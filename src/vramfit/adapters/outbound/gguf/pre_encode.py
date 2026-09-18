r"""The pre-encoding stage of the pack path (ADR-0032 decision 1).

Selected ``Q2_0`` tensors take vramfit's own assisted encoder before
stock ``llama-quantize`` runs. This module owns the pack side of
that stage, and it imports no torch (ADR-0005, ADR-0008): it selects
the tensors, refuses the cases the quantizer would abort on before
anything is written, drives the encoder as a separate program, and
verifies the payloads the packed file carries. The encoder itself
lives in the scan adapter package
([vramfit.adapters.outbound.scan.q2_0_program][]), and the pack
adapter launches it under the interpreter that carries torch, the
way it already runs ``convert_hf_to_gguf.py``.

**Selection follows the quantizer's own matching.** A tensor is a
candidate when some override the pack drives resolves to ``q2_0``
and matches its name the way ``llama-quantize`` matches: the pattern
lower-cased and searched, not anchored. ``llama-quant.cpp`` applies
the first matching pattern, so the candidate's *first* match must be
``q2_0`` too. When an earlier override wins with another type, the
quantizer would meet an existing ``Q2_0`` tensor at a different
resolved type and exit 1 with ``requantizing from type q2_0 is
disabled``, after the preprocessor already wrote a full-size file.
ADR-0032's consequence on ADR-0012 decision 3 rules that this
refuses before the preprocessor writes, and this module refuses
there. A candidate whose rows do not divide into 64-element blocks
refuses for the same reason: ``tensor_type_fallback`` would
substitute a type, and the substituted type is a different resolved
type.

**Three cases pack unassisted, as before.** Under a method that
reads the matrix, a tensor the matrix does
not cover, one the recipe's exclusions drop, and one the quantizer
never quantizes (a norm, a router, a one-dimensional tensor, or a
name a dedicated flag binds) stay float in the mixed file, and the
stock pass fits them (ADR-0032 decision 3). The quantizer's skip
rules are ``tensor_allows_quantization`` in ``llama-quant.cpp`` at
b10362, and the four this module models are the ones a
``blk\.<n>\.`` pattern can reach. The selection drops these tensors
before it refuses. The preprocessor never writes one as ``Q2_0``,
so neither abort above can reach it, and refusing it would refuse a
pack the stock path serves.

**An unassisted method drops the first two.** Under ``q0-fit2``
the encoder reads no matrix at any width, so coverage and the
recipe's imatrix exclusions decide nothing about the nominal-2 fit
(ADR-0018, 2026-09-17 amendment). The selection then takes every
candidate the quantizer would quantize, which is what that method
prices. The skip rule and both refusals are unchanged.

**Verification reads the packed file's bytes.** After the quantizer
exits 0, every pre-encoded tensor must hold the type id ``42`` and
the exact payload bytes the encoder reported. A mismatch names the
tensor and keeps the file for inspection. Matching hashes protect
against lost pre-encoding and prove nothing about the fit; the
encoder's reference fixtures carry that.

Examples:
    The pack adapter drives the stage between its checks and the
    quantizer:

    ```python
    targets = select_pre_encode_targets(overrides, header, ...)
    ```

See Also:
    - [vramfit.adapters.outbound.gguf.mixed_gguf][]: The rewrite that
      places the payloads into the temporary mixed GGUF.
    - [vramfit.adapters.outbound.gguf.pack][]: The caller.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from vramfit.adapters.outbound.gguf.header import GgufHeader, TensorInfo, read_header
from vramfit.adapters.outbound.gguf.override_match import (
    EMBEDDING_TARGETS,
    OUTPUT_TARGETS,
    compiled_override,
)
from vramfit.adapters.outbound.gguf.q2_0_blocks import (
    Q2_0_TYPE_ID,
    QK2_0,
    q2_0_payload_bytes,
)
from vramfit.adapters.outbound.gguf.toolrun import run_tool
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.pack import TypeOverride

# The tensor-type name the ADR-0028 table emits for nominal 2 on a
# row the super-block does not divide. The only type the encoder
# pre-encodes.
Q2_0_TYPE_NAME: Final[str] = "q2_0"

# The quantizer skips a tensor under two dimensions, one whose name
# does not end in ``weight``, a norm, and the expert router
# (``tensor_allows_quantization``, llama-quant.cpp at b10362).
_QUANTIZED_SUFFIX: Final[str] = "weight"
_SKIPPED_NAME_PARTS: Final[tuple[str, ...]] = ("_norm.weight", "ffn_gate_inp.weight")
_MIN_QUANTIZED_DIMS: Final[int] = 2

# The interpreter bootstrap that starts the encoder program. The
# import happens in the child process only, so the pack path stays
# torch-free (ADR-0008).
ENCODER_BOOTSTRAP: Final[str] = (
    "import sys; from vramfit.adapters.outbound.scan.q2_0_program import main; "
    "sys.exit(main())"
)


@dataclass(frozen=True, slots=True)
class PreEncodeTarget:
    """One tensor the preprocessor rewrites.

    Attributes:
        name (str): The GGUF tensor name.
        elements (int): The tensor's element count.

    Examples:
        Size the payload the encoder will write:

        ```python
        size = q2_0_payload_bytes(target.elements)
        ```
    """

    name: str
    elements: int


@dataclass(frozen=True, slots=True)
class EncodedTensor:
    """What the encoder reported for one tensor.

    Attributes:
        name (str): The GGUF tensor name.
        payload (Path): The payload file the encoder wrote.
        sha256 (str): SHA-256 of the payload bytes.
        size (int): The payload's byte count.

    Examples:
        The verification compares the packed bytes to ``sha256``.
    """

    name: str
    payload: Path
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class EncoderReport:
    """The encoder program's report for one run.

    Attributes:
        encoder (str): The encoder revision that produced the
            payloads.
        tensors (tuple[EncodedTensor, ...]): One entry per target,
            in target order.

    Examples:
        Map the payloads into the mixed-GGUF rewrite:

        ```python
        payloads = {t.name: (t.payload, Q2_0_TYPE_ID) for t in report.tensors}
        ```
    """

    encoder: str
    tensors: tuple[EncodedTensor, ...]


def quantizer_skips(
    info: TensorInfo, *, embedding_flag: bool, output_flag: bool
) -> bool:
    """Tell whether stock ``llama-quantize`` never quantizes a tensor.

    Args:
        info: The tensor's header entry.
        embedding_flag: Whether the pack emits
            ``--token-embedding-type``, which binds the embedding
            before any pattern.
        output_flag: Whether the pack emits ``--output-tensor-type``,
            which binds the output head before any pattern.

    Returns:
        True when the quantizer copies the tensor at its stored
        type, so pre-encoding it would change what the stock pass
        ships.

    Examples:
        A norm never quantizes:

        ```python
        info = TensorInfo("blk.0.attn_norm.weight", (64,), 0, 0)
        assert quantizer_skips(info, embedding_flag=False, output_flag=False)
        ```
    """
    if len(info.dims) < _MIN_QUANTIZED_DIMS:
        return True
    if not info.name.endswith(_QUANTIZED_SUFFIX):
        return True
    if any(part in info.name for part in _SKIPPED_NAME_PARTS):
        return True
    if embedding_flag and info.name in EMBEDDING_TARGETS:
        return True
    return output_flag and info.name in OUTPUT_TARGETS


def select_pre_encode_targets(
    overrides: Sequence[TypeOverride],
    header: GgufHeader,
    *,
    covered: Collection[str] | None,
    excluded: Collection[str],
    embedding_flag: bool = False,
    output_flag: bool = False,
) -> tuple[PreEncodeTarget, ...]:
    """Name the tensors the preprocessor rewrites, refusing before it writes.

    Args:
        overrides: The overrides the pack drives, in priority order.
        header: The base GGUF's parsed header.
        covered: The imatrix's entry names. A candidate outside it
            packs unassisted. None for an unassisted method, which
            keeps every candidate and ignores ``excluded`` — neither
            names anything its fit reads.
        excluded: The recipe's imatrix exclusions. A candidate in it
            packs unassisted.
        embedding_flag: Whether the pack emits
            ``--token-embedding-type``.
        output_flag: Whether the pack emits ``--output-tensor-type``.

    Returns:
        The targets in header order. Empty when no ``q2_0`` override
        reaches a covered tensor.

    Raises:
        PackError: If a selected tensor's first matching override
            carries another type, or its rows do not divide into
            ``QK2_0`` blocks. Either would abort the quantizer
            mid-pack on a tensor the preprocessor already wrote, so
            the refusal comes first. An uncovered or excluded tensor
            drops out before both checks, because the preprocessor
            leaves it float.

    Examples:
        One covered stack at ``q2_0`` selects one target:

        ```python
        targets = select_pre_encode_targets(
            overrides, header, covered={"blk.0.ffn_down_exps.weight"}, excluded=()
        )
        ```
    """
    patterns = [(compiled_override(o), o) for o in overrides]
    q2_0_patterns = [
        pattern
        for pattern, override in patterns
        if override.quant_type == Q2_0_TYPE_NAME
    ]
    targets: list[PreEncodeTarget] = []
    for info in header.tensors:
        if not any(pattern.search(info.name) for pattern in q2_0_patterns):
            continue
        if quantizer_skips(
            info, embedding_flag=embedding_flag, output_flag=output_flag
        ):
            continue
        if covered is not None and (info.name not in covered or info.name in excluded):
            continue
        winner = next(
            override for pattern, override in patterns if pattern.search(info.name)
        )
        if winner.quant_type != Q2_0_TYPE_NAME:
            raise PackError(
                f'a q2_0 override reaches "{info.name}", but the quantizer '
                f'applies the first matching pattern, "{winner.pattern}" at '
                f"{winner.quant_type}. A pre-encoded Q2_0 tensor at another "
                "resolved type makes llama-quantize exit 1 with 'requantizing "
                "from type q2_0 is disabled', so the pack refuses before the "
                "preprocessor writes (ADR-0032)"
            )
        if info.dims[0] % QK2_0:
            raise PackError(
                f'"{info.name}" resolves to q2_0 with rows of {info.dims[0]}, '
                f"which do not divide into {QK2_0}-element blocks. The "
                "quantizer would fall back to another type, so the pack "
                "refuses before the preprocessor writes (ADR-0028, ADR-0032)"
            )
        targets.append(PreEncodeTarget(info.name, info.elements))
    return tuple(targets)


def _read_report(path: Path, targets: Sequence[PreEncodeTarget]) -> EncoderReport:
    """Read and vouch the encoder's report.

    Args:
        path: The report file the encoder wrote.
        targets: The tensors the encoder was asked for.

    Returns:
        The report: the encoder revision, and one payload path,
        size, and digest per target, in target order.

    Raises:
        PackError: If the report is missing, malformed, or does not
            name every target with a payload of the expected size.
    """
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackError(
            f"pre-encode: cannot read the encoder report {path}: {exc}"
        ) from exc
    try:
        entries = raw["tensors"]
        tensors = tuple(
            EncodedTensor(
                name=target.name,
                payload=Path(entries[target.name]["payload"]),
                sha256=str(entries[target.name]["sha256"]),
                size=int(entries[target.name]["bytes"]),
            )
            for target in targets
        )
        report = EncoderReport(str(raw["encoder"]), tensors)
    except (KeyError, TypeError, ValueError) as exc:
        raise PackError(
            f"pre-encode: the encoder report {path} is malformed: {exc}"
        ) from exc
    for target, tensor in zip(targets, report.tensors, strict=True):
        expected = q2_0_payload_bytes(target.elements)
        if tensor.size != expected or not tensor.payload.is_file():
            raise PackError(
                f'pre-encode: the encoder reported {tensor.size} bytes for "{target.name}", '
                f"expected {expected} at {tensor.payload}"
            )
    return report


def run_encoder(
    command: Sequence[str],
    *,
    base_gguf: Path,
    imatrix: Path | None,
    targets: Sequence[PreEncodeTarget],
    work_dir: Path,
    threads: int,
) -> EncoderReport:
    """Run the encoder program over the targets.

    Args:
        command: The program's argument vector prefix, normally the
            torch interpreter and `ENCODER_BOOTSTRAP`.
        base_gguf: The floating-point base the encoder reads.
        imatrix: The importance matrix that weights the fit, or
            None to fit every block at weight 1.0.
        targets: The tensors to encode.
        work_dir: Where the payloads and the report land. Created
            when absent.
        threads: The CPU thread count the encoder may use.

    Returns:
        The encoder's report, vouched against the targets.

    Raises:
        PackError: If the program cannot start, exits nonzero, or
            reports something other than one sized payload per
            target. The message carries the program's last lines.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / "report.json"
    argv = [
        *command,
        "--base-gguf",
        str(base_gguf),
        *(() if imatrix is None else ("--imatrix", str(imatrix))),
        "--out-dir",
        str(work_dir),
        "--report",
        str(report_path),
        "--threads",
        str(threads),
    ]
    for target in targets:
        argv += ["--tensor", target.name]
    run_tool(argv, stage="pre-encode")
    return _read_report(report_path, targets)


def verify_pre_encoded(packed: Path, expected: Mapping[str, str]) -> None:
    """Prove the packed file carries every pre-encoded payload unchanged.

    Args:
        packed: The file the quantizer wrote.
        expected: SHA-256 per pre-encoded tensor name, from the
            encoder's report.

    Raises:
        PackError: If the packed file cannot be read, a tensor is
            missing, holds a type other than ``Q2_0``, or its bytes
            hash differently. The file is kept for inspection.

    Examples:
        The pack adapter runs this right after the quantizer exits 0:

        ```python
        verify_pre_encoded(out_path, {t.name: t.sha256 for t in report.tensors})
        ```
    """
    header = read_header(packed)
    by_name = {info.name: info for info in header.tensors}
    try:
        with packed.open("rb") as handle:
            for name, digest in expected.items():
                info = by_name.get(name)
                if info is None:
                    raise PackError(
                        f'the packed file {packed} carries no tensor "{name}", '
                        "which the preprocessor pre-encoded (ADR-0032)"
                    )
                if info.type_id != Q2_0_TYPE_ID:
                    raise PackError(
                        f'the packed file {packed} holds "{name}" at type id '
                        f"{info.type_id}, not Q2_0 ({Q2_0_TYPE_ID}). The pre-encoded "
                        "payload was lost, and the file is kept for inspection "
                        "(ADR-0032)"
                    )
                handle.seek(header.data_start + info.offset)
                payload = handle.read(q2_0_payload_bytes(info.elements))
                found = hashlib.sha256(payload).hexdigest()
                if found != digest:
                    raise PackError(
                        f'the packed file {packed} holds different bytes for "{name}" '
                        f"than the preprocessor wrote (sha256 {found} versus {digest}). "
                        "The file is kept for inspection (ADR-0032)"
                    )
    except OSError as exc:
        raise PackError(f"cannot read the packed file {packed}: {exc}") from exc
