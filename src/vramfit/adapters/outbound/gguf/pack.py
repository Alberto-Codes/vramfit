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
layer group, plus dedicated embedding and output-head flags. It then
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
as amended 2026-09-04, #413, #414).

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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from vramfit.adapters.outbound.gguf.exclusion_match import check_exclusion_match
from vramfit.adapters.outbound.gguf.file_type import stamp_modal_file_type
from vramfit.adapters.outbound.gguf.override_match import check_base_coverage
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
from vramfit.domain.model import Recipe
from vramfit.domain.pack import PackResult

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
        python_bin (Path): Interpreter for the convert script. Must
            import torch.
        threads (int): Quantizer thread count.
        imatrix (Path | None): Importance matrix file for the
            quantizer (ADR-0016). None packs without one.

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
                the reader could not decode (#252), or the packed
                file cannot take its file type (#414).
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
        overrides = all_overrides(recipe)
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
        command += [str(self.base_gguf), str(self.out_path), base, str(self.threads)]
        quantize_output = run_tool(command, stage="quantize")
        # A type-fallback warning means the artifact ignored the
        # recipe on a zero exit — halt, never record-and-continue
        # (ADR-0028 decision 3).
        rewritten = tuple(_TYPE_FALLBACK.findall(quantize_output))
        if rewritten:
            raise TypeFallbackError(rewritten, self.out_path)
        # An excluded tensor's row is gone from the loaded matrix, so
        # the quantizer reports it as a miss — an intentional one,
        # recorded in imatrix_excluded instead (ADR-0023).
        uncovered = (
            tuple(
                name
                for name in _read_miss_names(quantize_output, self.out_path)
                if name not in excluded
            )
            if self.imatrix is not None
            else ()
        )
        packed_bytes = sized_file(self.out_path, stage="quantize")
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
        )
