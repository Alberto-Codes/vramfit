r"""Hold a CUDA ballast allocation so a run sees a smaller VRAM budget.

The RTX 4090 carries no hardware memory partition, so the 12 GiB serve
test runs on a full 24 GiB card under a ballast process (#164). One
process holds an allocation, and every other process sees the remainder.
A CUDA allocation lowers the Vulkan heap budget too, because both
backends draw on one physical pool.

The script calls the CUDA driver through ``ctypes``. It needs no torch.
ADR-0005 keeps torch behind the ``scan`` extra, and the serve test must
not pull the heavy stack into a path that does not need it. The script
imports the standard library only, so it runs under any Python 3.12 or
newer interpreter, inside the project venv or outside it.

The ballast is sized from **free** memory, never from total. The script
creates its CUDA context first and measures free memory after that, so
the context's own cost comes out of the ballast. The driver rounds the
allocation up, which leaves 1 to 2 MiB less visible than the target. The
printed number is the measured one, not the requested one.

The cap sets **visible free VRAM**, which is not the weight budget of
``docs/reference/glossary.md``. Visible free VRAM still has to cover the
KV cache and runtime overhead. A 12288 MiB cap holds a smaller weight
budget inside it.

Examples:
    Cap a llama.cpp device query at 12 GiB:

    ```console
    $ python3 scripts/vram_ballast.py --target-mib 12288 -- \\
        ~/quantfit-runs/llama-vulkan/llama-cli --list-devices
    ballast: device 0 free 22504 MiB, held 10216 MiB, visible 12286 MiB
    Available devices:
      Vulkan0: NVIDIA GeForce RTX 4090 (24564 MiB, 12286 MiB free)
    ```

    Hold the cap for a manual run, then release it with Ctrl-C:

    ```console
    $ python3 scripts/vram_ballast.py --target-mib 12288
    ```

Note:
    Send the signal to this script, not to a launcher that wraps it.
    ``uv run`` does not forward SIGTERM to the interpreter it starts, so
    a hold started that way outlives the signal and keeps the ballast.
"""

from __future__ import annotations

import argparse
import ctypes
import signal
import subprocess
import sys

MIB = 1024 * 1024
CUDA_SUCCESS = 0
DEFAULT_TARGET_MIB = 12288
DEFAULT_LIBRARY = "libcuda.so.1"
# The shell convention for a command that could not be executed.
EXIT_COMMAND_NOT_FOUND = 127
_WATCHED_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class CudaDriverError(RuntimeError):
    """The CUDA driver rejected a call, or the driver library is absent."""


class CudaBallast:
    """A CUDA context that holds one allocation until it is released.

    The class binds the driver entry points the ballast needs. Every
    binding declares ``argtypes``, because a byte count above 2 GiB
    truncates through the default ``c_int`` marshalling.
    """

    def __init__(self, library: str = DEFAULT_LIBRARY) -> None:
        """Load the CUDA driver and initialize it.

        Args:
            library: The driver shared object to load.

        Raises:
            CudaDriverError: If the library is absent or ``cuInit`` fails.
        """
        try:
            self._lib = ctypes.CDLL(library)
        except OSError as exc:
            raise CudaDriverError(f"cannot load {library}: {exc}") from exc
        self._declare_signatures()
        self._context: ctypes.c_void_p | None = None
        self._pointer: ctypes.c_ulonglong | None = None
        self._check(self._lib.cuInit(0), "cuInit")

    def _declare_signatures(self) -> None:
        """Pin argument and result types on every entry point used."""
        signatures = {
            "cuInit": [ctypes.c_uint],
            "cuGetErrorString": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
            "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
            "cuCtxCreate_v2": [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint,
                ctypes.c_int,
            ],
            "cuCtxDestroy_v2": [ctypes.c_void_p],
            "cuMemGetInfo_v2": [
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            ],
            "cuMemAlloc_v2": [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t],
            "cuMemFree_v2": [ctypes.c_ulonglong],
        }
        for name, argtypes in signatures.items():
            entry = getattr(self._lib, name)
            entry.argtypes = argtypes
            entry.restype = ctypes.c_int

    def _check(self, result: int, call: str) -> None:
        """Raise when a driver call returned anything but success.

        Args:
            result: The ``CUresult`` the call returned.
            call: The entry point name, for the message.

        Raises:
            CudaDriverError: On any non-success result.
        """
        if result == CUDA_SUCCESS:
            return
        message = ctypes.c_char_p()
        named = self._lib.cuGetErrorString(result, ctypes.byref(message))
        detail = (
            message.value.decode() if named == CUDA_SUCCESS and message.value else ""
        )
        raise CudaDriverError(f"{call} failed: {detail or f'CUresult {result}'}")

    def open_device(self, ordinal: int) -> None:
        """Create a CUDA context on one device.

        Args:
            ordinal: The device index.

        Raises:
            CudaDriverError: If the device or the context is unavailable.
        """
        device = ctypes.c_int()
        self._check(self._lib.cuDeviceGet(ctypes.byref(device), ordinal), "cuDeviceGet")
        context = ctypes.c_void_p()
        self._check(
            self._lib.cuCtxCreate_v2(ctypes.byref(context), 0, device),
            "cuCtxCreate",
        )
        self._context = context

    def memory_info(self) -> tuple[int, int]:
        """Report device memory as a ``(free, total)`` pair in bytes.

        Returns:
            Free and total bytes, as the driver sees them from this context.

        Raises:
            CudaDriverError: If the query fails.
        """
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        self._check(
            self._lib.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)),
            "cuMemGetInfo",
        )
        return free.value, total.value

    def hold(self, size_bytes: int) -> None:
        """Allocate the ballast and keep it until release.

        A zero size holds nothing. The driver rejects a zero-byte
        allocation, and free memory already sits at the target.

        Args:
            size_bytes: The allocation size.

        Raises:
            CudaDriverError: If the allocation fails.
        """
        if size_bytes == 0:
            return
        pointer = ctypes.c_ulonglong()
        self._check(
            self._lib.cuMemAlloc_v2(ctypes.byref(pointer), size_bytes),
            "cuMemAlloc",
        )
        self._pointer = pointer

    def release(self) -> None:
        """Free the allocation and destroy the context. Safe to repeat.

        Release runs on the way out of a failed run, so it reports a
        driver failure to stderr rather than raise over the original
        error. A failed free leaves the card short, and the operator
        needs to read that in the run log.
        """
        if self._pointer is not None:
            self._report(self._lib.cuMemFree_v2(self._pointer), "cuMemFree")
            self._pointer = None
        if self._context is not None:
            self._report(self._lib.cuCtxDestroy_v2(self._context), "cuCtxDestroy")
            self._context = None

    def _report(self, result: int, call: str) -> None:
        """Print a driver failure to stderr without raising."""
        if result == CUDA_SUCCESS:
            return
        print(
            f"error: {call} failed with CUresult {result} — the card may "
            "still hold the ballast",
            file=sys.stderr,
        )


def size_ballast(free_bytes: int, target_bytes: int) -> int:
    """Size the ballast from free memory, never from total.

    Args:
        free_bytes: Free device memory, measured after the context exists.
        target_bytes: The free VRAM the run must see.

    Returns:
        The ballast size in bytes.

    Raises:
        ValueError: If free memory does not already cover the target. The
            script refuses rather than cap to a smaller number in silence.
    """
    if target_bytes <= 0:
        raise ValueError(f"target must exceed 0 MiB, got {target_bytes // MIB} MiB")
    if free_bytes < target_bytes:
        raise ValueError(
            f"free VRAM is {free_bytes // MIB} MiB, below the "
            f"{target_bytes // MIB} MiB target — free the device or lower "
            "--target-mib"
        )
    return free_bytes - target_bytes


def run_command(command: list[str]) -> int:
    """Run the capped command and forward the watched signals to it.

    The handlers install before the child starts. A signal that arrives
    in between has no child to reach, so the handler records it and the
    function delivers it once the child exists. Installing the handlers
    after the child would leave a window where the default disposition
    kills this process and orphans an uncapped child.

    Args:
        command: The argument vector to execute.

    Returns:
        The child's wait status as a shell reports it: its exit code,
        128 plus the signal number when a signal killed it, or 127 when
        the command does not exist.
    """
    child: subprocess.Popen[bytes] | None = None
    pending: int | None = None

    def forward(number: int, _frame: object) -> None:
        nonlocal pending
        if child is None:
            pending = number
        else:
            child.send_signal(number)

    previous = {number: signal.signal(number, forward) for number in _WATCHED_SIGNALS}
    try:
        try:
            child = subprocess.Popen(command)  # noqa: S603 — the caller names it
        except (OSError, ValueError) as exc:
            print(f"error: cannot run {command[0]}: {exc}", file=sys.stderr)
            return EXIT_COMMAND_NOT_FOUND
        if pending is not None:
            child.send_signal(pending)
        return exit_status(child.wait())
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def exit_status(wait_result: int) -> int:
    """Convert a ``Popen.wait`` result to a shell exit status.

    ``wait`` returns the negated signal number when a signal killed the
    child. Passing that to ``SystemExit`` masks it to 241 for SIGTERM,
    where a shell reports 143. The harness runs under ``systemd-run``,
    which reads this status to decide why a unit failed.

    Args:
        wait_result: The value ``Popen.wait`` returned.

    Returns:
        The exit code, or 128 plus the signal number.
    """
    return 128 - wait_result if wait_result < 0 else wait_result


def wait_for_signal() -> int:
    """Hold the ballast until SIGINT or SIGTERM arrives.

    The mask is restored before returning. Release runs after this
    function, and ``cuCtxDestroy`` can stall while another process is
    busy on the device. A leftover mask would swallow the operator's
    second Ctrl-C and leave SIGKILL as the only way out.

    Returns:
        Zero. The signal is the intended way to end a manual hold.
    """
    watched = set(_WATCHED_SIGNALS)
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    try:
        caught = signal.sigwait(watched)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
    print(f"ballast: caught {signal.Signals(caught).name}, releasing", flush=True)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: The argument list, or None to read ``sys.argv``.

    Returns:
        The parsed namespace, with ``command`` stripped of a leading ``--``.
    """
    parser = argparse.ArgumentParser(
        prog="vram_ballast.py",
        description="Hold a CUDA ballast so a run sees less free VRAM.",
    )
    parser.add_argument(
        "--target-mib",
        type=int,
        default=DEFAULT_TARGET_MIB,
        help="Free VRAM the run must see, in MiB (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device ordinal (default: %(default)s).",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run under the cap. Omit it to hold until a signal.",
    )
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    """Take the cap, run the command or hold, then release.

    Args:
        argv: The argument list, or None to read ``sys.argv``.

    Returns:
        The child's exit code, or 1 when the cap cannot be taken.
    """
    args = parse_args(argv)
    target = args.target_mib * MIB
    try:
        ballast = CudaBallast()
        ballast.open_device(args.device)
    except CudaDriverError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        free, _ = ballast.memory_info()
        try:
            held = size_ballast(free, target)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            ballast.hold(held)
        except CudaDriverError as exc:
            print(
                f"error: cannot hold {held // MIB} MiB of {free // MIB} MiB "
                f"free: {exc}",
                file=sys.stderr,
            )
            return 1
        visible, _ = ballast.memory_info()
        print(
            f"ballast: device {args.device} free {free // MIB} MiB, "
            f"held {held // MIB} MiB, visible {visible // MIB} MiB",
            flush=True,
        )
        return run_command(args.command) if args.command else wait_for_signal()
    except CudaDriverError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        ballast.release()


if __name__ == "__main__":
    raise SystemExit(main())
