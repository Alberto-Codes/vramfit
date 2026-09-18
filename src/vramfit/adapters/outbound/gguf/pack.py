"""llama.cpp pack adapter: convert and quantize by subprocess.

Implements the `RecipePacker` port for the GGUF serving path
(ADR-0010, ADR-0012). `convert` runs ``convert_hf_to_gguf.py`` under
a caller-supplied interpreter — that interpreter carries torch, this
package never imports it (ADR-0005). `pack` first rejects a recipe
recorded for a foreign runtime (ADR-0013), then runs
``llama-quantize`` with the recipe's type mapping from
[vramfit.adapters.outbound.gguf.types][]: protection overrides
first (ADR-0022 — the quantizer applies the first matching
pattern), then pattern overrides per
layer group, plus dedicated embedding and output-head flags. That
mapping routes the 256 super-block decision from each group's
measured row width, which `checkpoint_row_widths` reads from the
checkpoint's shard headers (issue #515). It then
holds every override against the base GGUF's tensor names and refuses
one that matches nothing
([vramfit.adapters.outbound.gguf.override_match][]) — such an
override changes no type and the quantizer exits 0 without reporting
it (ADR-0012 as amended 2026-08-16). That one header read also holds
the embedding and output-head flags against the exact tensors they
bind, and refuses a scanned ``lm_head`` group the file carries no
``output.weight`` for — the head would take the ``--pure`` floor while
the record states the recipe's type (#306). The tied fallback stays
exempt, because decision 2 rules that flag a no-op. The read also names
the layers the file carries that no override reaches. They take the
``--pure`` floor, which decision 3 makes the designed outcome, so the
result records them and the pack continues (#307). The
adapter scans the quantizer's zero-exit output for the type-fallback
warning pair and halts on a match (`TypeFallbackError`) — a rewritten
type breaks the recipe the artifact claims to carry (ADR-0028). With
an importance matrix (ADR-0016) it also scans that output for
tensors the matrix did not cover — there the
quantizer only warns, and a silently unassisted tensor must not
pass unrecorded (ADR-0023 decision 4). A miss whose tensor name
carries U+FFFD halts
instead, because `run_tool` could not read that name and the
record would state a name nobody read (#252). That halt reports
stage ``quantize`` (ADR-0012 decision 5). The recipe's imatrix
exclusions become ``--exclude-weights`` flags, and their
intentional misses stay out
of that coverage record (ADR-0023). It holds those exclusions against
the matrix's entry names first and refuses one that reaches no row
([vramfit.adapters.outbound.gguf.exclusion_match][]) — the quantizer
erases nothing for such a name and exits 0, so the tensor would keep
the fit the recipe asked to drop (#309). Every failure — a tool that cannot start, exits
nonzero, dies to a signal, or leaves no usable file — translates to
`PackError` at this boundary (ADR-0011), carrying the tool's last
output lines. After a zero exit the adapter relabels the file: the
quantizer stamps ``general.file_type`` with the base ftype, and the
adapter rewrites it to the type covering the most bytes
([vramfit.adapters.outbound.gguf.file_type][], ADR-0012 decision 3
as amended 2026-09-04, #413, #414). A recipe priced with one of the
``Q2_0`` encoder's methods takes the pre-encoding path
(ADR-0032): the adapter selects the ``q2_0`` tensors the method
reaches — the matrix's covered ones under ``q0-imx2``, and every
candidate under unassisted ``q0-fit2`` (ADR-0018, 2026-09-17
amendment) —
refuses before anything is written when the quantizer's own matching
would floor one, runs the encoder as a separate program under
``python_bin``, writes the temporary mixed GGUF beside the output,
hands that file to the quantizer with the same flags and never
``--allow-requantize``, and verifies the packed payload bytes after
the zero exit ([vramfit.adapters.outbound.gguf.pre_encode][]). When
the stage itself fails, its temporaries go on a best-effort basis, so
a cleanup error never masks the failure being reported. A
failure inside that stage removes its temporaries. A failure after
it keeps the temporary mixed GGUF and the payload directory, and
names both.

Examples:
    Pack a recipe with a local llama.cpp checkout:

    ```python
    packer = LlamaCppPacker(
        model_dir=Path("model"),
        base_gguf=Path("model-f16.gguf"),
        out_path=Path("packed.gguf"),
        convert_script=Path("llama.cpp/convert_hf_to_gguf.py"),
        quantize_bin=Path("llama.cpp/build/bin/llama-quantize"),
        python_bin=Path(sys.executable),
    )
    packer.convert()
    result = packer.pack(recipe)
    ```

See Also:
    - [vramfit.ports.outbound][]: `RecipePacker`, which this
      satisfies.
"""

from __future__ import annotations

import contextlib
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from vramfit.adapters.outbound.gguf.exclusion_match import check_exclusion_match
from vramfit.adapters.outbound.gguf.file_type import stamp_modal_file_type
from vramfit.adapters.outbound.gguf.header import read_header
from vramfit.adapters.outbound.gguf.imatrix_counts import imatrix_entry_names
from vramfit.adapters.outbound.gguf.mixed_gguf import write_mixed_gguf
from vramfit.adapters.outbound.gguf.override_match import check_base_coverage
from vramfit.adapters.outbound.gguf.pre_encode import (
    ENCODER_BOOTSTRAP,
    EncoderReport,
    run_encoder,
    select_pre_encode_targets,
    verify_pre_encoded,
)
from vramfit.adapters.outbound.gguf.q2_0_blocks import Q2_0_TYPE_ID
from vramfit.adapters.outbound.gguf.toolrun import run_tool, sized_file
from vramfit.adapters.outbound.gguf.types import (
    PackError,
    all_overrides,
    base_type,
    check_runtime,
    imatrix_exclusion_names,
    output_group_type,
    output_tensor_type,
    token_embedding_type,
)
from vramfit.adapters.outbound.safetensors_sizes import SafetensorsSizes
from vramfit.domain.errors import VramfitError
from vramfit.domain.model import Q0_FIT2_METHOD, Q0_IMX2_METHOD, Recipe
from vramfit.domain.pack import PackResult, TypeOverride
from vramfit.domain.sizes import discovered_group_rows

# The quantizer's zero-exit warning for a tensor the importance
# matrix does not cover (llama.cpp src/llama-quant.cpp). The tensor
# is then quantized without importance data.
_IMATRIX_MISS: Final[re.Pattern[str]] = re.compile(r"did not find weights for (\S+)")

# The quantizer's zero-exit type-fallback warning pair
# (`tensor_type_fallback`, llama.cpp src/llama-quant.cpp): an
# ``ncols … not divisible`` report, then ``falling back to`` with the
# substituted type on the same output line. A rewritten type breaks
# the recipe the artifact claims to carry, so a match halts the pack
# (ADR-0028 decision 3).
_TYPE_FALLBACK: Final[re.Pattern[str]] = re.compile(
    r"warning: +(\S+) +- ncols +\d+ not divisible by +\d+ "
    r"\(required for type +(\S+)\).*?falling back to +(\S+)"
)

# `run_tool` replaces a byte it cannot decode with U+FFFD (#247).
# U+FFFD is not whitespace, so `_IMATRIX_MISS`'s capture group takes
# it like any other character (#252). The escape keeps the sentinel
# legible, because the glyph itself survives a re-encoding poorly.
_REPLACEMENT_CHAR: Final[str] = "\ufffd"


def _read_miss_names(output: str, out: Path) -> tuple[str, ...]:
    """Capture the imatrix-miss tensor names, refusing an unread one.

    The scan records a miss and continues (ADR-0028 decision 3), so
    a captured name reaches `PackResult.imatrix_uncovered` and the
    `model_packed` run-log event as fact. ADR-0023 decision 4 keeps
    that field an honest record of unintentional gaps. A name
    carrying U+FFFD was never read, so it records nothing. Dropping
    it silently would instead hide a gap the field must carry.
    Halting is the only answer that does neither.

    The ADR-0023 exclusion discount compares exact strings. A damaged
    name therefore evades it too. It then reads as a coverage gap the
    recipe meant to create.

    #252 measured the decode route to this case on 2026-08-15 and
    found it closed. llama.cpp truncates only inside its `- kv` dump
    loop, which no miss warning passes through. `ggml_set_name` cuts
    a name at 63 bytes, but the GGUF reader refuses an over-long
    name first (`ggml/src/gguf.cpp:639-644`), so that cut never runs
    here. The route that stays open is a GGUF file whose short
    tensor name is already invalid UTF-8, which no reader checks
    (`ggml/src/gguf.cpp:340-354`).

    The guard sees damage inside a captured name only. Damage to the
    surrounding literal deletes the match, and the miss then leaves
    no trace here. `run_tool` carries that residual case.

    Args:
        output: The quantizer's merged output.
        out: The packed file, kept for inspection.

    Returns:
        The miss names in output order, without repeats.

    Raises:
        PackError: If any captured name carries U+FFFD. The halt
            reports stage ``quantize`` (ADR-0012 decision 5), which
            also covers a quantizer that failed. #275 carries whether
            a zero-exit halt earns its own stage.

    Examples:
        A clean run reports its misses in order:

        ```python
        names = _read_miss_names(
            "did not find weights for blk.0.attn_v.weight", Path("out.gguf")
        )
        assert names == ("blk.0.attn_v.weight",)
        ```
    """
    names = tuple(dict.fromkeys(_IMATRIX_MISS.findall(output)))
    damaged = [name for name in names if _REPLACEMENT_CHAR in name]
    if damaged:
        # `ascii` spells the replacement character as an escape. A log
        # viewer that cannot render the glyph would otherwise show a
        # box in an error that is about that exact byte.
        details = ", ".join(ascii(name) for name in damaged)
        raise PackError(
            f"quantize: run_tool could not decode {len(damaged)} imatrix-miss "
            f"tensor name{'' if len(damaged) == 1 else 's'}: {details}. "
            f"An undecodable name records no imatrix coverage (ADR-0023). "
            f"The quantizer exited 0, and the packed file is kept at {out} "
            f"for inspection"
        )
    return names


class TypeFallbackError(PackError):
    """The quantizer rewrote tensor types the recipe assigned.

    Attributes:
        rewritten (tuple[tuple[str, str, str], ...]): One
            ``(tensor, requested_type, substituted_type)`` triple per
            rewritten tensor, in output order.

    Examples:
        The CLI reads the triples into the halt event:

        ```python
        for tensor, requested, substituted in exc.rewritten:
            ...
        ```
    """

    def __init__(self, rewritten: tuple[tuple[str, str, str], ...], out: Path) -> None:
        """Build the halt message from the parsed warning pairs.

        Args:
            rewritten: The parsed ``(tensor, requested, substituted)``
                triples.
            out: The packed file, kept for inspection.
        """
        details = ", ".join(
            f"{tensor}: {requested} -> {substituted}"
            for tensor, requested, substituted in rewritten
        )
        super().__init__(
            f"the quantizer rewrote {len(rewritten)} tensor type"
            f"{'' if len(rewritten) == 1 else 's'} the recipe assigned "
            f"(ADR-0028): {details}. The packed file no longer carries "
            f"the recipe and is kept at {out} for inspection"
        )
        self.rewritten = rewritten


def checkpoint_row_widths(model_dir: Path) -> Mapping[str, int]:
    """Measure each group's row width from a checkpoint's shard headers.

    The 256 super-block decision reads the width the checkpoint
    states, never a class name (issue #515). The pack reads it from
    the same source the plan priced from, so the predicted type and
    the emitted type cannot disagree. The read is a shard-header
    parse and needs no torch (ADR-0029 decision 1).

    The granularity is ``stack``: it fuses a routed-expert projection
    into the group a pack addresses and leaves every other tensor
    under its own name, which is the group set `tensor_overrides`
    maps.

    Args:
        model_dir: The Hugging Face checkpoint directory.

    Returns:
        Elements per row per group, under
        `vramfit.domain.sizes.MAP_ROOT`.

    Raises:
        PackError: If the checkpoint cannot be read or priced. The
            source's own message carries the reason.

    Examples:
        ```python
        widths = checkpoint_row_widths(Path("/models/qwen3-coder-30b"))
        ```
    """
    try:
        sizes = SafetensorsSizes(model_dir).tensor_sizes()
        return discovered_group_rows(sizes, "stack")
    except VramfitError as exc:
        raise PackError(
            f"cannot measure the checkpoint's row widths at {model_dir}: "
            f"{exc}. The 256 super-block decision reads those widths "
            f"(ADR-0028, issue #515)"
        ) from exc
    except OSError as exc:
        raise PackError(
            f"cannot read the checkpoint at {model_dir}: {exc}. The pack "
            f"measures each group's row width there (issue #515)"
        ) from exc


@dataclass(frozen=True, slots=True)
class LlamaCppPacker:
    """`RecipePacker` adapter driving the llama.cpp toolchain.

    Attributes:
        model_dir (Path): Hugging Face checkpoint directory.
        base_gguf (Path): Full-precision base GGUF. `convert` creates
            it when absent and reuses it when present.
        out_path (Path): Packed model destination.
        convert_script (Path): ``convert_hf_to_gguf.py`` path.
        quantize_bin (Path): ``llama-quantize`` path.
        python_bin (Path): Interpreter for the convert script and the
            assisted ``Q2_0`` encoder program (ADR-0032). Must import
            vramfit and torch.
        threads (int): Quantizer thread count.
        imatrix (Path | None): Importance matrix file for the
            quantizer (ADR-0016). None packs without one.
        row_widths (Mapping[str, int]): Elements per row per group,
            from `checkpoint_row_widths` over ``model_dir``. The 256
            super-block decision reads this width (issue #515), and
            the composition root passes the same mapping the plan
            priced from.
        encoder_command (tuple[str, ...] | None): The argument vector
            that starts the assisted ``Q2_0`` encoder program
            (ADR-0032). None runs `ENCODER_BOOTSTRAP` under
            ``python_bin``, which must import vramfit and torch.
            Tests substitute a stub here. The stage writes the mixed
            GGUF at `mixed_gguf` and the payloads under
            `pre_encode_dir`, both beside ``out_path``.

    Examples:
        The composition root wires the paths:

        ```python
        packer: RecipePacker = LlamaCppPacker(
            model_dir,
            base_gguf,
            out_path,
            convert_script,
            quantize_bin,
            python_bin,
        )
        ```
    """

    model_dir: Path
    base_gguf: Path
    out_path: Path
    convert_script: Path
    quantize_bin: Path
    python_bin: Path
    threads: int = 8
    imatrix: Path | None = None
    row_widths: Mapping[str, int] = field(default_factory=dict)
    encoder_command: tuple[str, ...] | None = None

    @property
    def mixed_gguf(self) -> Path:
        """Name the temporary mixed GGUF the preprocessor writes.

        Returns:
            The path beside ``out_path``.
        """
        return self.out_path.with_name(self.out_path.stem + ".mixed.gguf")

    @property
    def pre_encode_dir(self) -> Path:
        """Name the working directory the encoder writes payloads into.

        Returns:
            The directory beside ``out_path``.
        """
        return self.out_path.with_name(self.out_path.stem + ".pre-encode")

    def _pre_encode(
        self,
        recipe: Recipe,
        overrides: tuple[TypeOverride, ...],
        excluded: tuple[str, ...],
        *,
        embedding_flag: bool,
        output_flag: bool,
    ) -> EncoderReport | None:
        """Run the pre-encoding stage when the recipe's method selects it.

        Args:
            recipe: The recipe to pack.
            overrides: The composed overrides, in priority order.
            excluded: The recipe's imatrix exclusions.
            embedding_flag: Whether the pack emits
                ``--token-embedding-type``.
            output_flag: Whether the pack emits ``--output-tensor-type``.

        Returns:
            The encoder's report, or None when the recipe takes the
            stock path or no covered ``q2_0`` tensor exists.

        Raises:
            PackError: If the recipe's method needs an imatrix the
                pack lacks, the selection refuses (ADR-0032, ADR-0018's
                2026-09-17 amendment), the
                encoder fails, or the mixed file cannot be written.
                A failure after the encoder starts removes the
                payload directory and the mixed GGUF, and the
                message names both.
        """
        if recipe.within_group not in (Q0_IMX2_METHOD, Q0_FIT2_METHOD):
            return None
        assisted = recipe.within_group == Q0_IMX2_METHOD
        covered: frozenset[str] | None = None
        if assisted:
            if self.imatrix is None:
                raise PackError(
                    f'the recipe was priced with method "{recipe.within_group}", '
                    "whose Q2_0 cells the assisted encoder fitted, and the pack has "
                    "no --imatrix. Without the matrix every Q2_0 tensor would ship "
                    "stock, which the recipe did not price (ADR-0032 decision 3)"
                )
            covered = frozenset(imatrix_entry_names(self.imatrix))
        targets = select_pre_encode_targets(
            overrides,
            read_header(self.base_gguf),
            covered=covered,
            excluded=excluded,
            embedding_flag=embedding_flag,
            output_flag=output_flag,
        )
        if not targets:
            return None
        command = self.encoder_command or (
            str(self.python_bin),
            "-c",
            ENCODER_BOOTSTRAP,
        )
        try:
            report = run_encoder(
                command,
                base_gguf=self.base_gguf,
                imatrix=self.imatrix if assisted else None,
                targets=targets,
                work_dir=self.pre_encode_dir,
                threads=self.threads,
            )
            payloads = {
                tensor.name: (tensor.payload, Q2_0_TYPE_ID) for tensor in report.tensors
            }
            write_mixed_gguf(self.base_gguf, self.mixed_gguf, payloads)
        except PackError as exc:
            # The stage's own failure ships no artifact, and its
            # temporaries are multi-gigabyte. A full disk is one way
            # to reach here, so they go (ADR-0032).
            self._discard_pre_encode_files()
            raise PackError(
                f"{exc}\nThe pre-encoding stage removed its temporary files: the "
                f"payload directory {self.pre_encode_dir} and the mixed GGUF "
                f"{self.mixed_gguf} (ADR-0032)"
            ) from exc
        return report

    def _kept_temporaries(self) -> str:
        """Name the stage temporaries a later failure keeps.

        Returns:
            The sentence naming the temporary mixed GGUF and the
            payload directory, both kept for inspection.
        """
        return (
            f"The temporary mixed GGUF {self.mixed_gguf} and the payload "
            f"directory {self.pre_encode_dir} are kept for inspection (ADR-0032)"
        )

    def _discard_pre_encode_files(self) -> None:
        """Remove the temporary mixed GGUF and the payload directory.

        Best effort: the cleanup runs on a failure path, and a
        cleanup error must never replace the failure being reported.
        Whatever cannot be removed stays where it is.
        """
        with contextlib.suppress(OSError):
            self.mixed_gguf.unlink(missing_ok=True)
        shutil.rmtree(self.pre_encode_dir, ignore_errors=True)

    def _quantize(
        self,
        command: list[str],
        excluded: tuple[str, ...],
        stage: EncoderReport | None,
    ) -> tuple[tuple[str, ...], int]:
        """Run the quantizer and vouch for the file it wrote.

        Args:
            command: The quantizer's argument vector.
            excluded: The recipe's imatrix exclusions.
            stage: The pre-encoding stage's report, or None when the
                recipe took the stock path.

        Returns:
            The imatrix-miss tensor names and the packed file's size.

        Raises:
            PackError: If the quantizer fails, writes no usable file,
                names a miss the reader could not decode, or dropped
                a pre-encoded payload. After a pre-encoding stage the
                message names the temporaries it keeps.
            TypeFallbackError: If the zero-exit output carries the
                type-fallback warning pair (ADR-0028).
        """
        try:
            output = run_tool(command, stage="quantize")
            # A type-fallback warning means the artifact ignored the
            # recipe on a zero exit — halt, never record-and-continue
            # (ADR-0028 decision 3).
            rewritten = tuple(_TYPE_FALLBACK.findall(output))
            if rewritten:
                raise TypeFallbackError(rewritten, self.out_path)
            # An excluded tensor's row is gone from the loaded matrix,
            # so the quantizer reports it as a miss — an intentional
            # one, recorded in imatrix_excluded instead (ADR-0023).
            uncovered = (
                tuple(
                    name
                    for name in _read_miss_names(output, self.out_path)
                    if name not in excluded
                )
                if self.imatrix is not None
                else ()
            )
            packed_bytes = sized_file(self.out_path, stage="quantize")
            if stage is not None:
                # Matching bytes prove the stock pass copied every
                # pre-encoded tensor (ADR-0032 decision 1). The
                # temporary files go only after that proof.
                verify_pre_encoded(
                    self.out_path,
                    {tensor.name: tensor.sha256 for tensor in stage.tensors},
                )
        except PackError as exc:
            if stage is None:
                raise
            exc.args = (f"{exc}\n{self._kept_temporaries()}",)
            raise
        return uncovered, packed_bytes

    def convert(self) -> int:
        """Materialize the f16 base GGUF, reusing any existing file.

        The convert tool runs through the shared toolchain plumbing
        ([vramfit.adapters.outbound.gguf.toolrun][]).

        Returns:
            Size of the base GGUF in bytes.

        Raises:
            PackError: If the convert tool fails, writes no usable
                file, or the file cannot be inspected.
        """
        if not self.base_gguf.exists():
            run_tool(
                [
                    str(self.python_bin),
                    str(self.convert_script),
                    str(self.model_dir),
                    "--outfile",
                    str(self.base_gguf),
                    "--outtype",
                    "f16",
                ],
                stage="convert",
            )
        return sized_file(self.base_gguf, stage="convert")

    def pack(self, recipe: Recipe) -> PackResult:
        """Quantize the base GGUF into the recipe's packed model.

        The embedding and output-head flags resolve independently: an
        ``lm_head`` group drives the output flag with its own
        assignment, and the embedding assignment stands in when the
        scan measured no head (ADR-0012). A protected recipe's
        resolved pairs become the leading overrides (ADR-0022). Every
        override must match a tensor the base GGUF carries, and one
        that matches nothing refuses before the quantizer runs
        (#303). Each dedicated flag must reach its exact target
        tensor, and a scanned ``lm_head`` group against a file with no
        ``output.weight`` refuses there too (#306). The tied fallback
        does not, because decision 2 rules that flag a no-op. The same
        read names the layers the file carries that
        no override reaches. They take the ``--pure`` floor, so the
        result records them and the pack continues (#307). A
        configured importance
        matrix reaches the quantizer as ``--imatrix``, lands in the
        result's provenance, and the quantizer's output is scanned
        for tensors the matrix did not cover (ADR-0016). The
        recipe's imatrix exclusions become ``--exclude-weights``
        flags (ADR-0023) — the quantizer drops those rows, so the
        coverage scan discounts them as intentional. An exclusion the
        matrix carries no row for refuses before the quantizer runs,
        because it would erase nothing and report nothing (#309).
        Without an imatrix the exclusions are no-ops and stay
        unemitted. After the quantizer exits 0 and the fallback scan
        passes, the packed file's ``general.file_type`` becomes the
        modal type by bytes, and the result records it (ADR-0012
        decision 3 as amended 2026-09-04).

        The override composition reads each group's measured row
        width, which `checkpoint_row_widths` supplies to the
        composition root. The 256 super-block decision reads that
        width and never a class name (issue #515).

        A recipe priced with one of vramfit's ``Q2_0`` encoder
        methods pre-encodes its ``q2_0`` tensors first (ADR-0032).
        ``q0-imx2`` pre-encodes the tensors the matrix covers, and
        unassisted ``q0-fit2`` every candidate, because its fit
        reads no matrix (ADR-0018, 2026-09-17 amendment). The
        stage refuses before it writes when the quantizer's own
        matching would leave such a tensor at another type, runs the
        encoder as a separate program, writes the temporary mixed
        GGUF beside the output, and hands that file to the quantizer
        under the same flags. ``--allow-requantize`` is never passed.
        After the zero exit the packed payload bytes must match what
        the encoder wrote, and only then do the temporary files go.
        A failure inside the stage removes them at once, because no
        artifact depends on them. A failure after it keeps both the
        temporary mixed GGUF and the payload directory, and the
        message names them.
        The result records the pre-encoded tensors, the encoder
        revision, and whether the matrix weighted that fit.

        Args:
            recipe: The recipe to apply.

        Returns:
            The accounting record, with the real packed size and the
            resolved flag types.

        Raises:
            PackError: If the recipe targets another runtime
                (ADR-0013), the base GGUF is missing, the recipe
                cannot be mapped (ADR-0012), an override matches no
                tensor in the base GGUF (#303), a dedicated flag
                reaches no target tensor there (#306), an exclusion
                reaches no imatrix row (#309), the quantizer fails, it writes
                no usable file, it names an imatrix-miss tensor
                the reader could not decode (#252), the packed
                file cannot take its file type (#414), the recipe's
                method needs an imatrix the pack lacks, the
                pre-encoding selection refuses, the encoder fails,
                or a pre-encoded payload did not survive the
                quantizer (ADR-0032).
            TypeFallbackError: If the quantizer's output carries the
                type-fallback warning pair — the artifact ignored
                the recipe on a zero exit (ADR-0028). The file is
                kept for inspection.
        """
        check_runtime(recipe)
        if not self.base_gguf.exists():
            raise PackError(
                f"base GGUF {self.base_gguf} does not exist — run convert first"
            )
        base = base_type(recipe)
        embedding = token_embedding_type(recipe)
        output = output_tensor_type(recipe)
        # Protection overrides first (ADR-0022): the quantizer applies
        # the first matching pattern, so a protected tensor must match
        # its own pattern before its group's.
        overrides = all_overrides(recipe, self.row_widths)
        # Last mapping check, and the only one that reads the model.
        # An override the base GGUF carries no tensor for changes no
        # type, and the quantizer reports nothing and exits 0. It runs
        # after the table lookups above, so a recipe that fails both
        # still reports the table error first (ADR-0012 as amended
        # 2026-08-16, #303). The same read holds the two dedicated
        # flags against their target tensors (#306), and names the
        # layers the file carries that no override reaches, which take
        # the --pure floor on a zero exit (#307).
        #
        # Only a scanned lm_head group makes the output flag
        # load-bearing. Without one the flag carries the embedding's
        # type, and decision 2 already rules that it never applies on
        # a tied model — a ruled no-op, not a malformed input.
        floored_layers = check_base_coverage(
            overrides,
            self.base_gguf,
            embedding_flag=embedding is not None,
            output_flag=output_group_type(recipe) is not None,
        )
        excluded: tuple[str, ...] = ()
        if self.imatrix is not None:
            excluded = imatrix_exclusion_names(recipe)
            # The quantizer erases an imatrix row by substring and
            # reports no exclusion that erased nothing (#309). It runs
            # after the base-GGUF read, so a recipe that fails both
            # still reports the mapping error first.
            check_exclusion_match(excluded, self.imatrix)
        # The pre-encoding stage (ADR-0032). It refuses before it
        # writes, and it runs after every cheaper check so a recipe
        # that fails both still reports the mapping error first.
        stage = self._pre_encode(
            recipe,
            overrides,
            excluded,
            embedding_flag=embedding is not None,
            output_flag=output_group_type(recipe) is not None,
        )
        source = self.base_gguf if stage is None else self.mixed_gguf
        command = [str(self.quantize_bin), "--pure"]
        if self.imatrix is not None:
            command += ["--imatrix", str(self.imatrix)]
        for name in excluded:
            command += ["--exclude-weights", name]
        if embedding is not None:
            command += ["--token-embedding-type", embedding]
        if output is not None:
            # Without the flag an untied output head would fall to the
            # --pure base type — the recipe's floor (ADR-0012).
            command += ["--output-tensor-type", output]
        for override in overrides:
            command += ["--tensor-type", f"{override.pattern}={override.quant_type}"]
        command += [str(source), str(self.out_path), base, str(self.threads)]
        uncovered, packed_bytes = self._quantize(command, excluded, stage)
        pre_encoded: tuple[str, ...] = ()
        encoder: str | None = None
        if stage is not None:
            self._discard_pre_encode_files()
            pre_encoded = tuple(tensor.name for tensor in stage.tensors)
            encoder = stage.encoder
        # The quantizer stamped the base ftype, which names the floor
        # and not the file (#413). Relabel with the modal type by
        # bytes (ADR-0012 decision 3 as amended 2026-09-04).
        declared = stamp_modal_file_type(self.out_path)
        return PackResult(
            packed_bytes=packed_bytes,
            base_type=base,
            token_embedding_type=embedding,
            output_tensor_type=output,
            overrides=overrides,
            imatrix_path=None if self.imatrix is None else str(self.imatrix),
            imatrix_uncovered=uncovered,
            imatrix_excluded=excluded,
            floored_layers=floored_layers,
            file_type=declared,
            pre_encoded=pre_encoded,
            q2_0_encoder=encoder,
            pre_encode_assisted=recipe.within_group == Q0_IMX2_METHOD
            and bool(pre_encoded),
        )
