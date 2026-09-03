r"""Backfill per-tensor sizes into a sensitivity map (ADR-0022).

Reads each safetensors shard header of a model checkpoint — a JSON
parse of the first bytes, no torch — and writes an annotated copy of
the map with ``tensor_bytes`` on every group (the map-copy mechanism
from ADR-0021 decision 4). Sizes are element counts at 2 bytes per
parameter, matching the scan's bf16 reference convention.

The script reports each refusal below as ``error:`` on stderr and exits
1. Every refusal names the file it read or wrote, except the two that
compare a map against a checkpoint and name the group instead. It
refuses a document whose ``groups`` it cannot read (#298). It refuses a
map whose tensors the checkpoint does not cover. It refuses a group
whose checkpoint sizes disagree with its recorded ``bytes_fp16``. It
refuses two shards that define one tensor name (#297). It refuses an
``--out`` naming the input map or a checkpoint shard. It refuses a path
it cannot resolve, a directory it cannot list, a file it cannot open,
and an ``--out`` it cannot write or render. It refuses a map it cannot
parse and a shard it cannot parse. It refuses a shard shorter than the
8-byte length prefix. It refuses a shard that declares a header longer
than the file holds. It refuses a directory holding no shards. It
refuses a header record it cannot price. It refuses a JSON document that
defines the same key twice, under the rule #262 set for every vramfit
reader. Both reads apply that rule: the map and each shard header. It
refuses a document holding an integer outside the signed 64-bit range
(#317). That is the bound `_save_json` applies before every artifact
write, so this writer emits nothing the map reader's bound refuses.

Each refusal this module renders quotes at most `NAME_LIMIT` characters
of a tensor or group name, and states an oversized integer by bit width.
A publisher writes those names. No record sets that bound. ADR-0011's
2026-08-16 amendment applies the same principle to one envelope clause
and reaches no further, so the cap is this script's own practice. One
message escapes it: `DuplicateKeyError.message` embeds the repeated key
whole, and five readers share it. #363 carries that one.

One consequence survives the write guard. ``write_text`` truncates on
open, so a write that fails part-way leaves a short file at ``--out``.
The refusal names that path. Discard the file it names.

The shard header reader lives in
`vramfit.adapters.outbound.safetensors_sizes` and this script imports
it (ADR-0029 decision 1). Its refusals are that module's, so a header
refusal cannot differ between this script and `plan`. Every refusal
below the shard header is this script's own.

The ``groups`` guard reads that key and no more of the envelope. A
document carrying a readable ``groups`` and no ``vramfit_schema``,
``model_id``, or ``scan`` annotates here and still fails
`vramfit.adapters.outbound.sensitivity_map_json.map_from_dict`. That
reader owns the envelope. #335 carries whether this guard widens.

Examples:
    Annotate an existing map:

    ```console
    $ uv run python scripts/backfill_tensor_sizes.py \\
        sensitivity.json /models/nemotron --out sensitivity-sized.json
    backfilled 82 groups (438 tensors) -> sensitivity-sized.json
    ```
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# The four artifact writers import `_save_json` from this module the
# same way. The walk is shared rather than copied, so the script and
# the map reader cannot disagree on the bound (#317).
from vramfit.adapters.outbound.json_common import (
    ArtifactError,
    _check_writable_ints,
)
from vramfit.adapters.outbound.json_duplicate_key import (
    DuplicateKeyError,
    object_from_pairs,
)
from vramfit.adapters.outbound.safetensors_sizes import read_safetensors_header

# The longest name a refusal quotes. A header key is publisher-written
# and unbounded. Before this cap the group-coverage refusal rendered
# 200,080 bytes of stderr against a pair of 100,000-character names
# (#335). No record fixes the width, so 80 is a choice.
NAME_LIMIT = 80


class Refusal(ValueError):
    """The script cannot read an input, or must not write an output.

    `main` catches `ValueError` to print one ``error:`` line and exit 1,
    so this subclass reaches that path unchanged. It exists because many
    guards behind it test types, and a bare `ValueError` there fights
    ruff's `TRY004`. Raising `TypeError` instead would force `main` to
    catch `TypeError` as well, and that wider clause would relabel a
    real one as an operator mistake.

    One class covers the map and the checkpoint. Nothing catches a
    refusal by type, so a second class would name a boundary no code
    reads.
    """


def bounded(name: str) -> str:
    """Cut one tensor name to the length a refusal may quote.

    Args:
        name: A tensor name from a publisher-written header.

    Returns:
        The name, or its first `NAME_LIMIT` characters and an ellipsis.
    """
    if len(name) <= NAME_LIMIT:
        return name
    return f"{name[:NAME_LIMIT]}..."


def render_int(value: int) -> str:
    """State one integer, or its width when it will not convert.

    `int.__str__` raises past `sys.get_int_max_str_digits`, 4300 by
    default. A shard declaring a 4290-digit dimension priced a product
    that would not render, and the refusal leaked the interpreter's own
    message naming no file (#335).

    Args:
        value: An integer bound for a refusal message.

    Returns:
        The decimal digits, or a bit width when they exceed the limit.
    """
    try:
        return str(value)
    except ValueError:
        return f"an integer of {value.bit_length()} bits"


def resolve_path(path: Path) -> Path:
    """Resolve one path argument, refusing what the filesystem cannot.

    `Path.resolve` raises `RuntimeError` on a symlink loop, and that sits
    outside the `ValueError` clause `main` catches. The comparison it
    feeds ran above the try block, so a looping ``--out`` reached a
    traceback (#335).

    Args:
        path: A path from the command line.

    Returns:
        The resolved path.

    Raises:
        Refusal: If the path does not resolve.
    """
    try:
        return path.resolve()
    except (OSError, RuntimeError) as exc:
        raise Refusal(f"{path}: cannot resolve: {exc}") from exc


def read_map_document(path: Path) -> object:
    """Parse the input map, naming the file for every failure.

    `read_safetensors_header` converts each parse failure with the shard
    path. This read did not, so a malformed map reached a bare traceback
    (#287, #335). The parse clauses match
    `vramfit.adapters.outbound.hf_config.shape_from_config_json`, down to
    the catch-all below the two named subclasses.

    Args:
        path: The sensitivity map to annotate.

    Returns:
        The parsed document, of whatever JSON type the file holds.

    Raises:
        Refusal: If the file does not open, is not UTF-8, is not valid
            JSON, nests past the recursion limit, or defines the same key
            twice.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal(f"{path}: cannot read: {exc.strerror}") from exc
    except UnicodeDecodeError as exc:
        raise Refusal(f"{path}: not valid UTF-8: {exc}") from exc
    try:
        return json.loads(text, object_pairs_hook=object_from_pairs)
    except DuplicateKeyError as exc:
        raise Refusal(f"{path}: {exc.message}") from exc
    except json.JSONDecodeError as exc:
        raise Refusal(f"{path}: invalid JSON: {exc}") from exc
    except RecursionError as exc:
        # Deep nesting exhausts the decoder's stack. `RecursionError`
        # is no `ValueError`, so it escaped every clause here (#335).
        raise Refusal(f"{path}: JSON nests too deeply: {exc}") from exc
    except ValueError as exc:
        # An integer literal past `sys.get_int_max_str_digits` (4300 by
        # default) fails here, below the subclass clause above (#287).
        # `DuplicateKeyError` is no `ValueError`, so the structural
        # refusal cannot land here whatever the order.
        raise Refusal(f"{path}: cannot parse JSON: {exc}") from exc


def record_tensor_bytes(record: object, name: str, shard: Path) -> int:
    """Price one header record at the bf16 reference.

    A publisher writes the header, so the record is untrusted like the
    document around it. `math.prod` took whatever ``shape`` held. A
    string shape multiplied into a longer string. The group sum then
    raised a `TypeError` several frames away.

    Every dimension must reach 1, and the cases differ. The safetensors
    format types a shape as ``Vec<usize>``, so a negative dimension and
    a non-integer one cannot occur in a valid file. A zero dimension
    can, because the format permits a zero-element tensor and tests it.
    This script refuses that too: the tensor prices at zero bytes, and
    `sensitivity_map_json.py:485` requires a positive ``tensor_bytes``
    value. Refusing here beats writing a map the project's own reader
    rejects, and the operator learns at the backfill rather than at the
    next read. The maintainer ruled the early refusal on 2026-08-19,
    and `docs/reference/sensitivity-map.md` carries it under
    ``tensor_bytes``.

    `bool` subclasses `int`, so a ``true`` dimension counted as 1 until
    the integer test came first.

    The refusal names the offending Python type and never the value, and
    `bounded` caps the name beside it.

    Args:
        record: One header entry, of whatever JSON type the file holds.
        name: The tensor the entry describes.
        shard: The file it came from, for the refusal message.

    Returns:
        ``numel * 2`` bytes.

    Raises:
        Refusal: If the record is not an object, carries no ``shape``,
            or holds a dimension below 1.
    """
    where = f'{shard}: tensor "{bounded(name)}"'
    if not isinstance(record, dict):
        raise Refusal(f"{where}: record is not an object")
    if "shape" not in record:
        raise Refusal(f'{where}: no "shape"')
    shape = record["shape"]
    if not isinstance(shape, list):
        raise Refusal(f'{where}: "shape" is not a list')
    for dim in shape:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise Refusal(
                f'{where}: "shape" holds a {type(dim).__name__}, not a dimension'
            )
        if dim < 1:
            raise Refusal(f'{where}: "shape" holds a dimension below 1')
    return math.prod(shape) * 2


def checkpoint_shards(model_dir: Path) -> list[Path]:
    """List the checkpoint's safetensors shards.

    `Path.glob` swallows every `OSError`, so an absent directory and an
    unreadable one both reported "no shards found". The operator read
    that the checkpoint holds nothing when the process could not list it
    (#335). `iterdir` raises instead, and the message says which.

    Args:
        model_dir: Checkpoint directory holding ``*.safetensors``.

    Returns:
        Every shard, sorted by path.

    Raises:
        Refusal: If the directory does not list, or holds no shards.
    """
    try:
        entries = sorted(model_dir.iterdir())
    except OSError as exc:
        raise Refusal(f"{model_dir}: cannot list: {exc.strerror}") from exc
    shards = [path for path in entries if path.match("*.safetensors")]
    if not shards:
        raise Refusal(f"{model_dir}: no *.safetensors shards found")
    return shards


def checkpoint_tensor_bytes(shards: list[Path]) -> dict[str, int]:
    """Collect every checkpoint tensor's size at the bf16 reference.

    Two shards that declare one tensor name mean two shapes at once, so
    the script cannot price the tensor. It refuses rather than keeping
    whichever shard sorts last (#297). An equal repeated size refuses
    too, matching `object_from_pairs`, which reads the repeat and never
    the values.

    Args:
        shards: The checkpoint's shards, from `checkpoint_shards`.

    Returns:
        Mapping of tensor name to ``numel * 2`` bytes.

    Raises:
        Refusal: If two shards define one tensor name, or a shard does
            not read.
    """
    sizes: dict[str, int] = {}
    origin: dict[str, Path] = {}
    for shard in shards:
        for name, record in read_safetensors_header(shard).items():
            if name in origin:
                raise Refusal(
                    f'{shard}: tensor "{bounded(name)}" is already defined by '
                    f"{origin[name]} — two copies of one checkpoint here?"
                )
            sizes[name] = record_tensor_bytes(record, name, shard)
            origin[name] = shard
    return sizes


def _require_group_fields(group: dict, where: str) -> None:
    """Refuse a group the annotation loop cannot read.

    The loop reads three fields off every group. It reports a refusal
    under ``name``, it tests membership over ``tensors``, and it compares
    a sum against ``bytes_fp16``. A group missing one reaches a bare
    `KeyError`. A group carrying the wrong type is worse. A string
    ``tensors`` iterates per character, so the loop blames the checkpoint
    for a tensor the map never named.

    Args:
        group: One element of the map's ``groups`` list.
        where: The file and index, for the refusal message.

    Raises:
        Refusal: If a field is absent, or carries the wrong type.
    """
    for field in ("name", "tensors", "bytes_fp16"):
        if field not in group:
            raise Refusal(f'{where}: no "{field}" — not a sensitivity map')
    if not isinstance(group["name"], str):
        raise Refusal(f'{where}: "name" is not a string')
    if not isinstance(group["bytes_fp16"], int) or isinstance(
        group["bytes_fp16"], bool
    ):
        raise Refusal(f'{where}: "bytes_fp16" is not an integer')
    tensors = group["tensors"]
    if not isinstance(tensors, list) or not all(isinstance(t, str) for t in tensors):
        raise Refusal(f'{where}: "tensors" is not a list of names')
    if not tensors:
        raise Refusal(f'{where}: "tensors" is empty — a group names one tensor')


def map_groups(raw: object, path: Path) -> list[dict]:
    """Return the map's groups, refusing what the loop cannot read.

    The script reaches for ``groups`` and annotates each element. Given a
    recipe, it iterated zero times and wrote an unannotated copy on a
    zero exit (#298). That report also worded a genuinely empty map, so
    the two cases read alike. Handing the script the wrong artifact is
    the most likely operator mistake, so it refuses here.

    An empty ``groups`` list refuses too. `map_from_dict` already
    requires a map to carry one group, so a zero-group document is not a
    map the project accepts.

    This guard checks the fields the annotation loop dereferences and
    stops there. It reads no other envelope field, so it accepts a
    document that `map_from_dict` refuses.
    `vramfit.adapters.outbound.sensitivity_map_json` owns schema
    validation, and this script never grows a second copy of it.

    Args:
        raw: The parsed input document.
        path: The file it came from, for the refusal message.

    Returns:
        The ``groups`` list, every element a readable group.

    Raises:
        Refusal: If ``groups`` is absent, or the loop cannot read it.
    """
    if not isinstance(raw, dict) or "groups" not in raw:
        raise Refusal(f'{path}: not a sensitivity map — no "groups" key')
    groups = raw["groups"]
    if not isinstance(groups, list):
        raise Refusal(f'{path}: "groups" is not a list — not a sensitivity map')
    if not groups:
        raise Refusal(f'{path}: "groups" is empty — a map carries one group')
    for i, group in enumerate(groups):
        where = f"{path}: groups[{i}]"
        if not isinstance(group, dict):
            raise Refusal(f"{where} is not an object — not a sensitivity map")
        _require_group_fields(group, where)
    return groups


def annotate(groups: list[dict], sizes: dict[str, int], path: Path) -> int:
    """Write ``tensor_bytes`` on every group, refusing a wrong checkpoint.

    Both refusals name the map, the group, and the tensor. The scan
    reads group and tensor names off the checkpoint, so a publisher
    writes them and `bounded` caps what they render.

    Args:
        groups: The map's groups, each one readable.
        sizes: Every checkpoint tensor's size at the bf16 reference.
        path: The map the groups came from, for the refusal message.

    Returns:
        The number of tensors annotated.

    Raises:
        Refusal: If the checkpoint does not cover a group's tensors, or
            a group's sizes disagree with its ``bytes_fp16``.
    """
    tensor_count = 0
    for group in groups:
        where = f'{path}: group "{bounded(group["name"])}"'
        missing = [t for t in group["tensors"] if t not in sizes]
        if missing:
            raise Refusal(
                f"{where}: checkpoint has no tensor "
                f'"{bounded(missing[0])}" — wrong checkpoint for this map?'
            )
        tensor_bytes = {t: sizes[t] for t in group["tensors"]}
        total = sum(tensor_bytes.values())
        if total != group["bytes_fp16"]:
            raise Refusal(
                f"{where}: checkpoint sizes sum to {render_int(total)} but "
                f"the map records bytes_fp16 "
                f"{render_int(group['bytes_fp16'])} — wrong checkpoint for "
                "this map?"
            )
        group["tensor_bytes"] = tensor_bytes
        tensor_count += len(group["tensors"])
    return tensor_count


def same_file(out: Path, target: Path) -> bool:
    """Report whether ``--out`` and one input are the same file.

    A resolved path catches a second name reached by a symlink or by
    ``..``. It cannot catch a hardlink, which gives one inode two names
    in the same directory. `Path.samefile` compares the device and the
    inode, so it catches every case (#335).

    Args:
        out: The resolved ``--out`` path.
        target: An input file to compare against.

    Returns:
        True when both names reach one file.
    """
    if out == resolve_path(target):
        return True
    try:
        return out.samefile(target)
    except OSError:
        # `--out` names no file yet, which is the ordinary case.
        return False


def require_out_is_not_a_shard(out: Path, shards: list[Path]) -> None:
    """Refuse an ``--out`` that would destroy a checkpoint shard.

    The script already refused an ``--out`` naming the input map. It
    took one naming a shard: it read that shard, priced it, then
    overwrote it with JSON on a zero exit (#335). The shards are input
    too, and the guard runs before the write.

    Args:
        out: The resolved ``--out`` path.
        shards: The checkpoint's shards.

    Raises:
        Refusal: If ``--out`` names a shard.
    """
    for shard in shards:
        if same_file(out, shard):
            raise Refusal(f"--out must not name a checkpoint shard: {shard}")


def require_readable_ints(raw: object, path: Path) -> None:
    """Refuse a document outside the writer's integer bound.

    `_save_json` bounds every integer it writes to the signed 64-bit
    range, so vramfit reads what vramfit writes (ADR-0011's 2026-08-16
    amendment). This script wrote with a bare `json.dumps` and applied
    no bound (#317). Measured on a map recording a ``bytes_fp16`` of
    2 * 10^30 against a shard declaring that shape. The script wrote
    the copy and exited 0. `map_from_dict` then refused the copy at
    that field. The walk here is the one `_save_json` runs, so the two
    cannot disagree.

    The reach is narrow. `annotate` requires a group's sizes to sum to
    its ``bytes_fp16``, so an out-of-range size implies an out-of-range
    ``bytes_fp16`` in the input. The map reader already refuses that
    input. The script carried it forward, and now refuses it and names
    the map. The walk covers the whole document, so an integer the
    script never reads refuses too, under its own JSON path. That
    reaches a field the map reader does not know, which it would load
    with a warning. `_save_json` refuses it the same way.

    The refusal keeps the walk's ``$``-rooted path, so the message
    matches the one the map reader renders for the same field. A
    publisher-written key can be long, and `bounded` caps the path.

    The walk recurses in Python, so a document nested past the
    interpreter's limit raises `RecursionError` here before the render
    guard sees it. That is no `ValueError`, so it takes its own clause.

    Args:
        raw: The annotated document.
        path: The map it came from, for the refusal message.

    Raises:
        Refusal: If any integer is outside the signed 64-bit range, or
            the document nests past the recursion limit.
    """
    try:
        _check_writable_ints(raw, "$")
    except ArtifactError as exc:
        raise Refusal(f"{path}: {bounded(exc.json_path)}: {exc.message}") from exc
    except RecursionError as exc:
        raise Refusal(f"{path}: JSON nests too deeply: {exc}") from exc


def render_map(raw: object, out: Path) -> str:
    """Render the annotated map, refusing what the encoder cannot.

    `json.dumps` with an indent uses the pure-Python encoder, whose
    recursion budget is smaller than the C scanner's. So a document
    `read_map_document` parsed can still exhaust the encoder, and
    `RecursionError` is no `ValueError`. Measured on 3.12: a group
    carrying 1,200 levels of nesting reached a traceback where 20,000
    levels refused at the read (#335).

    The render runs before the write opens the file, so a refusal here
    leaves no output at all.

    Args:
        raw: The annotated document.
        out: The path the caller will write, for the refusal message.

    Returns:
        The document's JSON text, with a trailing newline.

    Raises:
        Refusal: If the document nests past the encoder's limit.
    """
    try:
        return json.dumps(raw, indent=2) + "\n"
    except RecursionError as exc:
        raise Refusal(f"{out}: cannot render: JSON nests too deeply: {exc}") from exc


def main() -> int:
    """Annotate a sensitivity map with per-tensor sizes.

    Returns:
        Process exit code: 0 on success, 1 on refusal.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_path", type=Path, help="sensitivity map to annotate")
    parser.add_argument("model_dir", type=Path, help="checkpoint with safetensors")
    parser.add_argument("--out", type=Path, required=True, help="annotated map copy")
    args = parser.parse_args()

    try:
        out = resolve_path(args.out)
        if same_file(out, args.map_path):
            raise Refusal("--out must differ from the input map")
        raw = read_map_document(args.map_path)
        groups = map_groups(raw, args.map_path)
        shards = checkpoint_shards(args.model_dir)
        require_out_is_not_a_shard(out, shards)
        sizes = checkpoint_tensor_bytes(shards)
        tensor_count = annotate(groups, sizes, args.map_path)
        require_readable_ints(raw, args.map_path)
        text = render_map(raw, args.out)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        args.out.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"error: {args.out}: cannot write: {exc.strerror}", file=sys.stderr)
        return 1
    print(f"backfilled {len(groups)} groups ({tensor_count} tensors) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
