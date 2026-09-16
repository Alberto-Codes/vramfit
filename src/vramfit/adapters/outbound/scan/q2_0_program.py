"""The preprocessor's encoder program (ADR-0032 decisions 1 and 2).

The pack path stays torch-free, so the assisted ``Q2_0`` encoder
runs as a separate program under the interpreter that carries torch,
the way ``convert_hf_to_gguf.py`` already does. This module is that
program. [vramfit.adapters.outbound.gguf.pre_encode][] launches it
with the base GGUF, the importance matrix, the tensor names to
encode, and a working directory. For each tensor it reads the
floating-point rows from the base, resolves the imatrix rows the way
``llama-quant.cpp`` slices them per expert, runs the one shared fit
in [vramfit.adapters.outbound.scan.q2_0_assisted][], and streams the
stored blocks into a payload file. It then writes a JSON report the
pack side reads back: the encoder revision, and one payload path,
size, and SHA-256 per tensor.

The imatrix read is `imatrix.load_imatrix`, the scan meter's own
reader, so the zero-count fallback and the per-expert row mapping
are the ones the meter applies (decision 3). Rows are encoded in
bounded chunks, and the base's tensor data is a memory map, so the
peak memory is one chunk's float32 workspace plus the imatrix.

Examples:
    The pack side runs the program like this:

    ```console
    $ python -c "..." --base-gguf model-f16.gguf --imatrix m.gguf
        --out-dir work --report work/report.json --threads 8
        --tensor blk.0.ffn_down_exps.weight
    ```

See Also:
    - [vramfit.adapters.outbound.gguf.pre_encode][]: The caller, and
      the report's reader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from gguf import GGUFReader

from vramfit.adapters.outbound.scan.imatrix import ImatrixEntry, load_imatrix
from vramfit.adapters.outbound.scan.q0_ref import QK2_0
from vramfit.adapters.outbound.scan.q2_0_assisted import (
    Q2_0_ENCODER_REVISION,
    q2_0_encode_rows,
)

# Rows per fit call. At the 2688-wide target rows this is a 44 MiB
# float32 workspace per chunk, plus the fit's temporaries.
_CHUNK_ROWS: Final[int] = 4096
# A fused expert stack reads as (experts, rows, columns).
_STACK_DIMS: Final[int] = 3
_SUPPORTED_DTYPES: Final[tuple[str, ...]] = ("float16", "float32")


class EncodeError(RuntimeError):
    """The program cannot encode what it was asked for.

    Examples:
        The program prints the message and exits 1:

        ```python
        raise EncodeError("tensor x is not in the base GGUF")
        ```
    """


def _find_tensor(reader: Any, name: str, base_gguf: Path) -> Any:
    """Locate one tensor in the open base.

    Args:
        reader: An open ``GGUFReader``.
        name: The tensor name.
        base_gguf: The base file, for the message.

    Returns:
        The reader's tensor entry.

    Raises:
        EncodeError: If the base carries no such tensor, or holds it
            at a type the program cannot read as float.
    """
    for tensor in reader.tensors:
        if tensor.name == name:
            if tensor.data.dtype.name not in _SUPPORTED_DTYPES:
                raise EncodeError(
                    f'"{name}" in {base_gguf} holds {tensor.data.dtype.name} data, '
                    "and the encoder reads float16 or float32 rows"
                )
            return tensor
    raise EncodeError(f'"{name}" is not in the base GGUF {base_gguf}')


def _weights_for(
    entry: ImatrixEntry, name: str, shape: tuple[int, ...]
) -> torch.Tensor:
    """Vouch the imatrix entry against the tensor's shape.

    Args:
        entry: The entry `load_imatrix` read for ``name``.
        name: The tensor name.
        shape: The tensor's numpy shape, experts first on a stack.

    Returns:
        Column weights, shape ``(matrices, row)``: one row per
        expert on a stack, one row for a dense tensor.

    Raises:
        EncodeError: If the entry's matrix count does not fit the
            tensor, or its columns do not match the row width.
    """
    matrices = int(entry.counts.numel())
    experts = shape[0] if len(shape) == _STACK_DIMS else 1
    if matrices != experts:
        raise EncodeError(
            f'"{name}" holds {experts} matrices and its imatrix entry counts '
            f"{matrices}. The imatrix does not describe this base"
        )
    weights = entry.column_weights
    if int(weights.shape[-1]) != shape[-1]:
        raise EncodeError(
            f'imatrix weights for "{name}" have {int(weights.shape[-1])} columns, '
            f"the tensor rows have {shape[-1]}"
        )
    return weights.to(torch.float32)


def _encode_tensor(
    tensor: Any, weights: torch.Tensor, payload: Path
) -> tuple[int, str]:
    """Stream one tensor's blocks into its payload file.

    Args:
        tensor: The reader's tensor entry, whose ``data`` is a
            memory-mapped float array shaped experts first.
        weights: Column weights, shape ``(matrices, row)``.
        payload: The file to write.

    Returns:
        The payload's byte count and SHA-256.

    Raises:
        EncodeError: If the rows do not divide into blocks.
    """
    data = tensor.data
    row = int(data.shape[-1])
    if row % QK2_0:
        raise EncodeError(
            f'"{tensor.name}" has rows of {row}, which do not divide into '
            f"{QK2_0}-element Q2_0 blocks"
        )
    matrices = data.reshape(weights.shape[0], -1, row)
    digest = hashlib.sha256()
    written = 0
    with payload.open("wb") as handle:
        for index in range(matrices.shape[0]):
            qw = weights[index : index + 1]
            for start in range(0, matrices.shape[1], _CHUNK_ROWS):
                chunk = np.ascontiguousarray(
                    matrices[index, start : start + _CHUNK_ROWS]
                )
                rows = torch.from_numpy(chunk.astype(np.float32, copy=False))
                blocks = q2_0_encode_rows(rows, qw.expand(rows.shape[0], row))
                handle.write(blocks)
                digest.update(blocks)
                written += len(blocks)
    return written, digest.hexdigest()


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the program's arguments.

    Args:
        argv: The arguments, or None for ``sys.argv``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="vramfit-q2-0-encoder",
        description="Pre-encode assisted Q2_0 tensors from a base GGUF (ADR-0032).",
    )
    parser.add_argument("--base-gguf", type=Path, required=True)
    parser.add_argument("--imatrix", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tensor", action="append", default=[], dest="tensors")
    return parser.parse_args(argv)


def encode(args: argparse.Namespace) -> dict[str, Any]:
    """Encode every requested tensor and build the report.

    Args:
        args: The parsed arguments.

    Returns:
        The report as a JSON-ready mapping: the encoder revision,
        and one payload path, size, digest, and element count per
        tensor.

    Raises:
        EncodeError: If a tensor is missing, unreadable, uncovered
            by the imatrix, or misshaped against its entry.
        ValueError: If the imatrix file is not one, per
            `load_imatrix`.
        OSError: If a file cannot be read or written.
    """
    torch.set_num_threads(max(1, int(args.threads)))
    reader = GGUFReader(str(args.base_gguf))
    entries = load_imatrix(args.imatrix)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, Any] = {}
    for index, name in enumerate(args.tensors):
        tensor = _find_tensor(reader, name, args.base_gguf)
        entry = entries.get(name)
        if entry is None:
            raise EncodeError(
                f'the imatrix {args.imatrix} carries no entry for "{name}"'
            )
        weights = _weights_for(entry, name, tuple(int(d) for d in tensor.data.shape))
        payload = args.out_dir / f"{index:04d}.q2_0"
        size, digest = _encode_tensor(tensor, weights, payload)
        tensors[name] = {
            "payload": str(payload),
            "bytes": size,
            "sha256": digest,
            "elements": int(tensor.data.size),
        }
    return {"encoder": Q2_0_ENCODER_REVISION, "tensors": tensors}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the program.

    Args:
        argv: The arguments, or None for ``sys.argv``.

    Returns:
        0 on success, after printing the tensor count and the
        encoder revision. 1 on a refusal the message explains.

    Examples:
        Encode one tensor from a test harness:

        ```python
        code = main(["--base-gguf", "b.gguf", "--imatrix", "m.gguf", ...])
        ```
    """
    args = _parse(argv)
    try:
        report = encode(args)
        args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
    except (EncodeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"pre-encoded {len(report['tensors'])} tensor(s) with {report['encoder']}")
    return 0
