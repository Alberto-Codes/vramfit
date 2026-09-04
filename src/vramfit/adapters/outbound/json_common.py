"""Shared JSON validation machinery for the artifact adapters.

`ArtifactError` sits under the `VramfitError` root (ADR-0011).
Every extractor takes a JSON path string so validation errors read like
``$.groups[3].sensitivity: key "4x" is not an integer precision``.
Numeric extractors reject booleans (JSON ``true`` is a valid Python int)
and non-finite floats (``json.loads`` accepts ``NaN``/``Infinity``, which
would poison solver comparisons downstream). A number written as an
integer literal too large for a float fails as an `ArtifactError`, not
an `OverflowError` (#260). A literal past the parser's own digit limit
fails the same way, at the load step. An integer field bounds to the
signed 64-bit range, because Python integers are unbounded and a
document could otherwise declare a count no machine can hold. The
writer applies that bound too, so vramfit reads what vramfit writes.
A reader bounds what the format can carry and the domain bounds what
the value means (ADR-0008 as amended 2026-08-16). A document that repeats a key
inside one object fails at the load step too (#262). `json.loads` would
otherwise keep the last value and report nothing. That refusal names the
key and reports at the artifact root, because the parser hook that
catches it sees no ancestry. That hook lives in
`json_duplicate_key`, which three readers outside this module share
(#283). A field the reader does not know warns and loads
(#261) — `_warn_unknown_fields` reports it, and ADR-0013's
2026-08-16 amendment sets that level. An `UnknownFieldReporter`
carries the report — `report_through_warnings` by default, and the CLI
installs its own. The boolean extractor
accepts only real booleans, and the string extractors reject the
empty string. Schema versions advance
per artifact (ADR-0013) — each adapter owns its version constant and
passes it to `_check_schema_version`. An adapter reads one version
unless it names older ones through ``also_reads``, which suits a bump
that only widens what a document may say. Readers accept only the
post-rename envelope key (#118): a document carrying
``quantfit_schema`` fails with a message that names the new key and
the version this reader accepts (#154).

Examples:
    Report a validation failure with its JSON path:

    ```python
    from vramfit.adapters.outbound.json_common import ArtifactError

    try:
        raise ArtifactError("$.groups[3]", "expected a JSON object")
    except ArtifactError as exc:
        print(exc.json_path, exc.message)
    ```

See Also:
    - [vramfit.adapters.outbound.sensitivity_map_json][]: Map IO.
    - [vramfit.adapters.outbound.recipe_json][]: Recipe IO.
    - [vramfit.adapters.outbound.json_duplicate_key][]: The shared hook.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Final

from vramfit.adapters.outbound.json_duplicate_key import (
    DuplicateKeyError,
    object_from_pairs,
)
from vramfit.domain.errors import VramfitError


class ArtifactError(VramfitError, ValueError):
    """Invalid vramfit JSON artifact, under the `VramfitError` root.

    Attributes:
        json_path (str): Dotted path to the offending element, e.g.
            ``$.groups[3].sensitivity``.
        message (str): Human-readable description of the problem.

    Examples:
        Catch and report a validation failure:

        ```python
        from vramfit.adapters.outbound.json_common import ArtifactError

        try:
            _require(False, "$.scan", "missing")
        except ArtifactError as exc:
            print(exc.json_path, exc.message)
        ```
    """

    def __init__(self, json_path: str, message: str) -> None:
        """Build the error from a JSON path and a message.

        Args:
            json_path: Dotted path to the offending element.
            message: Human-readable description of the problem.
        """
        super().__init__(f"{json_path}: {message}")
        self.json_path = json_path
        self.message = message


def _require(condition: bool, path: str, message: str) -> None:
    """Raise `ArtifactError` at ``path`` unless ``condition`` holds.

    Args:
        condition: The invariant that must be true.
        path: JSON path reported on failure.
        message: Failure description.

    Raises:
        ArtifactError: If ``condition`` is false.
    """
    if not condition:
        raise ArtifactError(path, message)


class UnknownArtifactFieldWarning(UserWarning):
    """An artifact carries a field its reader does not know (#261).

    ADR-0013's 2026-08-16 amendment sets this level. The reader
    accepts the field and reports it. A save drops it, because no
    domain type carries it. The category exists so a caller can route
    or silence these reports alone — the CLI renders them on the human
    channel.

    Examples:
        Catch the report a hand-edited field raises. The default
        reporter must still be in force — the CLI replaces it:

        ```python
        import warnings

        from vramfit.adapters.outbound.evals_sidecar_json import (
            load_evals_sidecar,
        )
        from vramfit.adapters.outbound.json_common import (
            UnknownArtifactFieldWarning,
        )

        with warnings.catch_warnings(record=True) as seen:
            load_evals_sidecar(hand_edited_path)
        print(seen[0].category is UnknownArtifactFieldWarning)
        ```
    """


# Takes one report's text and delivers it. A plain comment, because no
# other alias in this package carries an attribute docstring.
UnknownFieldReporter = Callable[[str], None]


# A source location the reader can honestly claim. `warn_explicit`
# demands one, and every real candidate is wrong: the raising line sits
# inside this module, and the frame above it is a different depth at
# each of the 18 call sites. The JSON path inside the message is the
# locator that matters.
_ARTIFACT_ORIGIN: Final[str] = "<vramfit artifact>"

# The signed 64-bit range an artifact integer must fit (#260). Python
# integers are unbounded, so without this a document could declare a
# byte count no machine can hold and every reader would take it.
_INT_MAX: Final[int] = 2**63 - 1
_INT_MIN: Final[int] = -(2**63)


def report_through_warnings(message: str) -> None:
    """Deliver a report as an `UnknownArtifactFieldWarning`.

    The default. A caller that imports a reader and wires nothing still
    sees the report, and `warnings` filters silence or collect it.

    `warnings.warn` would drop most reports. Its default filter keys on
    the message, the category, the module, and the line, then records
    the hit in that module's registry. The raising line never moves, so
    the second document carrying the same field would report nothing.
    ``registry=None`` skips that bookkeeping, so every document reports.

    Args:
        message: The report text, already carrying its JSON path.

    Warns:
        UnknownArtifactFieldWarning: Always. The message is the report.
    """
    warnings.warn_explicit(
        message, UnknownArtifactFieldWarning, _ARTIFACT_ORIGIN, 0, registry=None
    )


# The reporter the readers call. A `ContextVar` rather than a rebound
# module name: `set_unknown_field_reporter` returns a token that
# restores the previous reporter exactly, and a thread or an asyncio
# task sees only its own installs.
_REPORTER: Final[ContextVar[UnknownFieldReporter]] = ContextVar(
    "vramfit_unknown_field_reporter", default=report_through_warnings
)


def set_unknown_field_reporter(
    report: UnknownFieldReporter,
) -> Token[UnknownFieldReporter]:
    """Route unknown-field reports somewhere else.

    The CLI installs a reporter that prints one ``warning:`` line,
    because the interpreter's own warning rendering names no artifact
    and tells an operator nothing.

    Args:
        report: The reporter to install.

    Returns:
        A token that `reset_unknown_field_reporter` restores from.
    """
    return _REPORTER.set(report)


def reset_unknown_field_reporter(token: Token[UnknownFieldReporter]) -> None:
    """Restore the reporter that a token was taken before.

    Args:
        token: The token `set_unknown_field_reporter` returned.
    """
    _REPORTER.reset(token)


def _warn_unknown_fields(obj: dict[str, Any], path: str, known: frozenset[str]) -> None:
    """Report every key in ``obj`` the reader does not know.

    The reader loads the document anyway (ADR-0013, the 2026-08-16
    amendment). Reports follow document order, so two runs over one
    file agree. An object whose keys the schema does not fix — the
    map's ``sensitivity``, the recipe's ``pins`` — never reaches this
    check. Its own rule validates its keys instead.

    The reporter in force takes each report, so a caller decides where
    it lands.

    Args:
        obj: The JSON object to check.
        path: JSON path of that object, for the report.
        known: Every key this reader carries at that object.
    """
    for key in obj:
        if key in known:
            continue
        _REPORTER.get()(
            f"{path}.{key}: vramfit does not know this field. A save drops it."
        )


def _get_dict(obj: Any, path: str) -> dict[str, Any]:
    """Return ``obj`` as a JSON object.

    Args:
        obj: Candidate value.
        path: JSON path for error reporting.

    Returns:
        The value, typed as a dict.

    Raises:
        ArtifactError: If the value is not a JSON object.
    """
    _require(isinstance(obj, dict), path, "expected a JSON object")
    return obj


def _get_list(obj: dict[str, Any], key: str, path: str) -> list[Any]:
    """Return the list stored at ``key``.

    Args:
        obj: Parent JSON object.
        key: Key to read.
        path: JSON path of the parent for error reporting.

    Returns:
        The list value.

    Raises:
        ArtifactError: If the key is missing or not a list.
    """
    _require(key in obj, path, f'missing required field "{key}"')
    value = obj[key]
    _require(isinstance(value, list), f"{path}.{key}", "expected a list")
    return value


def _get_str(obj: dict[str, Any], key: str, path: str) -> str:
    """Return the non-empty string stored at ``key``.

    Args:
        obj: Parent JSON object.
        key: Key to read.
        path: JSON path of the parent for error reporting.

    Returns:
        The string value.

    Raises:
        ArtifactError: If the key is missing, not a string, or empty.
    """
    _require(key in obj, path, f'missing required field "{key}"')
    value = obj[key]
    _require(isinstance(value, str), f"{path}.{key}", "expected a string")
    _require(value != "", f"{path}.{key}", "must not be empty")
    return value


def _as_str(value: Any, path: str) -> str:
    """Return ``value`` as a non-empty string.

    Args:
        value: Candidate value.
        path: JSON path for error reporting.

    Returns:
        The string value.

    Raises:
        ArtifactError: If the value is not a string, or empty.
    """
    _require(isinstance(value, str), path, "expected a string")
    _require(value != "", path, "must not be empty")
    return value


def _get_bool(obj: dict[str, Any], key: str, path: str) -> bool:
    """Return the boolean stored at ``key``.

    Args:
        obj: Parent JSON object.
        key: Key to read.
        path: JSON path of the parent for error reporting.

    Returns:
        The boolean value.

    Raises:
        ArtifactError: If the key is missing or not a boolean.
    """
    _require(key in obj, path, f'missing required field "{key}"')
    value = obj[key]
    _require(isinstance(value, bool), f"{path}.{key}", "expected a boolean")
    return value


def _as_int(value: Any, path: str) -> int:
    """Return ``value`` as an int, rejecting booleans and unbounded values.

    Python integers carry unlimited precision, so an artifact could
    declare a count or a byte size no machine can hold and every
    reader would record it as provenance. The bound is the signed
    64-bit range, which is what a byte count, a token count, and a
    chunk count all fit. The largest real value this project measures
    is a 93 GB checkpoint, near 10^11.

    The bound sits here rather than in the domain, matching
    `_as_float`: the reader answers whether the format can carry a
    value, and the domain answers whether the value means anything
    (#260). Above 4300 digits `_load_json` refuses the document
    first, so this closes the window between the two.

    Args:
        value: Candidate value.
        path: JSON path for error reporting.

    Returns:
        The integer value.

    Raises:
        ArtifactError: If the value is a bool, not an integer, or
            outside the signed 64-bit range.
    """
    _require(not isinstance(value, bool), path, "expected an integer, got a boolean")
    _require(isinstance(value, int), path, "expected an integer")
    _require(
        _INT_MIN <= value <= _INT_MAX,
        path,
        f"integer is outside the signed 64-bit range [{_INT_MIN}, {_INT_MAX}]",
    )
    return value


def _get_int(obj: dict[str, Any], key: str, path: str) -> int:
    """Return the integer stored at ``key``.

    Args:
        obj: Parent JSON object.
        key: Key to read.
        path: JSON path of the parent for error reporting.

    Returns:
        The integer value.

    Raises:
        ArtifactError: If the key is missing or not an integer.
    """
    _require(key in obj, path, f'missing required field "{key}"')
    return _as_int(obj[key], f"{path}.{key}")


def _as_float(value: Any, path: str) -> float:
    """Return ``value`` as a finite float, rejecting booleans and NaN/inf.

    Args:
        value: Candidate value.
        path: JSON path for error reporting.

    Returns:
        The numeric value as a float.

    Raises:
        ArtifactError: If the value is a bool, not a number, too large
            for a float, or not finite.
    """
    _require(not isinstance(value, bool), path, "expected a number, got a boolean")
    _require(isinstance(value, (int, float)), path, "expected a number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ArtifactError(path, "number is too large for a float") from exc
    _require(math.isfinite(result), path, "must be a finite number")
    return result


def _get_float(obj: dict[str, Any], key: str, path: str) -> float:
    """Return the number stored at ``key`` as a finite float.

    Args:
        obj: Parent JSON object.
        key: Key to read.
        path: JSON path of the parent for error reporting.

    Returns:
        The numeric value.

    Raises:
        ArtifactError: If the key is missing, not a number, too large
            for a float, or not finite.
    """
    _require(key in obj, path, f'missing required field "{key}"')
    return _as_float(obj[key], f"{path}.{key}")


_JSON_TYPE_NAMES: Final[dict[type, str]] = {
    bool: "boolean",
    str: "string",
    float: "number",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _pre_rename_version_clause(obj: dict[str, Any], readable: tuple[int, ...]) -> str:
    """Describe the version blocker a pre-rename artifact also carries.

    The key rename is one blocker. A version this reader cannot read
    is a second, and #154 set the rule that a message names both. A
    reader who fixes only the key otherwise meets the second one
    straight after.

    A non-integer version earns its own remedy (#260). "Bump the
    version" is advice no reader can follow when the value is
    ``"one"`` or ``true``, because there is no number to increment.
    The clause names the JSON type and never the value, so a document
    carrying a large object under that key cannot render an unbounded
    error message.

    Args:
        obj: Top-level artifact object.
        readable: Every schema version this adapter reads.

    Returns:
        The clause to append, or the empty string when the version
        needs none. A version this reader already accepts needs only
        the key rename.

    Examples:
        A readable version adds nothing:

        ```python
        assert _pre_rename_version_clause({"quantfit_schema": 2}, (2,)) == ""
        ```
    """
    if "quantfit_schema" not in obj:
        return ""
    value = obj["quantfit_schema"]
    if isinstance(value, bool) or not isinstance(value, int):
        name = _JSON_TYPE_NAMES.get(type(value), "value")
        return (
            f" The document declares a version of JSON type {name}. "
            "A key rename alone does not make it load. "
            "Write the version as an integer."
        )
    if value in readable:
        return ""
    return (
        f" The document declares version {value}. "
        "A key rename alone does not make it load. "
        "Bump the version or re-run the stage that writes it."
    )


def _reject_renamed_envelope_key(
    obj: dict[str, Any], path: str, readable: tuple[int, ...]
) -> None:
    """Reject an artifact that carries the pre-rename envelope key.

    The message names both blockers, not just the first (#154). Every
    artifact in the frozen run root (#134) predates the rename by one
    to four schema versions. A reader who renames the key alone fails
    again on the version, so the message states the version too.

    `_pre_rename_version_clause` builds the version half of the
    message, including the non-integer case this reader used to pass
    over in silence (#260).

    Args:
        obj: Top-level artifact object.
        path: JSON path of the artifact root.
        readable: Every schema version this adapter reads, in any
            order — membership decides the message and `sorted`
            renders it. A document already at one of these versions
            needs only the key rename, so the message must not also
            tell the reader to bump.

    Raises:
        ArtifactError: If the object carries ``quantfit_schema``. That
            key renamed to ``vramfit_schema`` with the tool (#118).
    """
    if "quantfit_schema" not in obj:
        return
    names = " or ".join(str(v) for v in sorted(readable))
    detail = _pre_rename_version_clause(obj, readable)
    raise ArtifactError(
        f"{path}.quantfit_schema",
        'the envelope key renamed to "vramfit_schema" (#118). '
        f"This vramfit reads only the new key at version {names}.{detail}",
    )


def _check_schema_version(
    obj: dict[str, Any],
    path: str,
    expected: int,
    also_reads: tuple[int, ...] = (),
) -> None:
    """Validate the artifact's ``vramfit_schema`` envelope field.

    Args:
        obj: Top-level artifact object.
        path: JSON path of the artifact root.
        expected: The schema version this artifact's adapter writes.
            Versions advance per artifact (ADR-0013) — the recipe
            writes 6 while the sensitivity map writes 3. Every
            caller passes its own constant, so no artifact silently
            validates against another's version.
        also_reads: Older versions this adapter still reads. Pass a
            version here only when its documents are already valid
            under ``expected`` — the sensitivity map reads 2 because
            version 3 only widened an enum (#161). Default empty, so
            an adapter reads one version until it states otherwise.

    Raises:
        ArtifactError: If the version is missing or unsupported — the
            message names every version this vramfit reads. A
            document carrying the pre-rename key gets the rename
            message instead, which names those versions as well as
            the key (#154).
    """
    readable = (expected, *also_reads)
    _reject_renamed_envelope_key(obj, path, readable)
    version = _get_int(obj, "vramfit_schema", path)
    names = " or ".join(str(v) for v in sorted(readable))
    _require(
        version in readable,
        f"{path}.vramfit_schema",
        f"unsupported schema version {version} — this vramfit reads version {names}",
    )


def _load_json(path: Path, root: str) -> dict[str, Any]:
    """Read ``path`` and return its top-level JSON object.

    Args:
        path: File to read.
        root: JSON path name for the artifact root in error messages.

    Returns:
        The parsed top-level object.

    Raises:
        ArtifactError: If the file cannot be read, is not UTF-8, is not
            valid JSON, carries a number literal the parser refuses,
            nests past the recursion limit (#478), repeats a key inside
            one object, or its top level is not an object. The
            duplicate-key message names the key, which is all
            `object_from_pairs` can report.
    """
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=object_from_pairs
        )
    except DuplicateKeyError as exc:
        raise ArtifactError(root, exc.message) from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(root, f"invalid JSON: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ArtifactError(root, f"not valid UTF-8: {exc}") from exc
    except RecursionError as exc:
        # Deep nesting exhausts the decoder's stack. `RecursionError`
        # is no `ValueError`, so it escaped every caller (#478).
        raise ArtifactError(root, f"JSON nests too deeply: {exc}") from exc
    except ValueError as exc:
        # An integer literal past `sys.get_int_max_str_digits` (4300 by
        # default) fails here, before any extractor sees it (#260). The
        # clause sits below the two ValueError subclasses above, which
        # carry their own messages.
        raise ArtifactError(root, f"cannot parse JSON: {exc}") from exc
    except OSError as exc:
        raise ArtifactError(root, f"cannot read file: {exc}") from exc
    return _get_dict(data, root)


def _check_writable_ints(value: Any, path: str) -> None:
    """Refuse an integer this project's own readers would not take.

    The bound in `_as_int` is one half of a round trip. Without the
    same bound here, a writer emits a document its own reader
    refuses, and the refusal names the artifact rather than the input
    that produced it (#260).

    Args:
        value: The value to walk. Objects and arrays recurse.
        path: JSON path of ``value`` for error reporting.

    Raises:
        ArtifactError: If any integer is outside the signed 64-bit
            range. Booleans are integers in Python and pass.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        _require(
            _INT_MIN <= value <= _INT_MAX,
            path,
            f"integer is outside the signed 64-bit range "
            f"[{_INT_MIN}, {_INT_MAX}] — no reader would load it back",
        )
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_writable_ints(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_writable_ints(item, f"{path}[{index}]")


def _save_json(data: dict[str, Any], path: Path) -> None:
    """Write ``data`` to ``path`` as pretty-printed JSON, atomically.

    The payload lands in a sibling temp file first and replaces the
    target in one step, so a failed write never leaves a truncated
    artifact behind.

    Every integer is held to the same bound the readers apply, so
    vramfit reads what vramfit writes (#260). The walk runs once over
    a document of a few thousand values.

    Args:
        data: JSON-serializable object.
        path: Destination file.

    Raises:
        ArtifactError: If any integer is outside the signed 64-bit
            range.
    """
    _check_writable_ints(data, "$")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
