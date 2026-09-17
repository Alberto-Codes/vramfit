"""The provenance mark: who established a recorded digest, and when.

One vocabulary serves every artifact that records a **content
identity**. The evals sidecar's corpus reference introduced it
(#589), and the sensitivity map's calibration file carries the same
mark. This module owns the three values and the two rules, so a
second artifact extends the mechanism instead of forking it.

`PROVENANCE_MARKS` holds the values. `check_mark_pairs_with_digest`
enforces that a digest and its mark arrive together. `check_referent`
enforces that a mark asserting about something outside the digest
names that referent — `recovered` names the file, `re_derived` names
the revision. Each caller passes the field names its own schema
spells, so one rule reports in the reader's own vocabulary.

Examples:
    Refuse a re-derived digest that names no revision:

    ```python
    from vramfit.domain.provenance import RE_DERIVED, check_referent

    check_referent(RE_DERIVED, file=None, revision=None)  # raises ValueError
    ```

See Also:
    - [vramfit.domain.model][]: `ScanMeta`, the sensitivity map's mark.
    - [vramfit.domain.evals][]: `CorpusReference`, the sidecar's mark.
"""

from __future__ import annotations

# The process that produced the numbers hashed these bytes as it read
# them. It asserts only about the bytes the digest already names.
MEASURED = "measured"
# The run's own file survived and was hashed afterwards: the same
# bytes at a later moment. It asserts that a file survived.
RECOVERED = "recovered"
# The run's own file is gone, and these are the pinned revision's
# bytes. It asserts a revision the bytes came from, which is an
# assumption being recorded, not a measurement being recovered.
RE_DERIVED = "re_derived"
# The three values a mark accepts, in the order the glossary lists
# them. Every artifact that records a content identity reads this
# tuple — the vocabulary has one home.
PROVENANCE_MARKS = (MEASURED, RECOVERED, RE_DERIVED)


def check_mark_pairs_with_digest(
    digest: str | None,
    mark: str | None,
    *,
    digest_field: str,
    mark_field: str,
) -> None:
    """Enforce that a digest and its mark arrive together.

    A digest says which bytes. A mark says who hashed them and when.
    Either alone records a claim no reader can check, so the two
    pair in both directions.

    Args:
        digest: The recorded SHA-256, or None.
        mark: The recorded provenance mark, or None.
        digest_field: The digest field's name, for the message.
        mark_field: The mark field's name, for the message.

    Raises:
        ValueError: If one is present without the other, or the mark
            is not one of `PROVENANCE_MARKS`.

    Examples:
        ```python
        from vramfit.domain.provenance import check_mark_pairs_with_digest

        check_mark_pairs_with_digest(
            "0" * 64,
            "measured",
            digest_field="sha256",
            mark_field="provenance",
        )
        ```
    """
    if (digest is None) != (mark is None):
        raise ValueError(
            f"{digest_field} and {mark_field} must pair — a digest says "
            "which bytes, and the mark says who hashed them and when"
        )
    if mark is not None and mark not in PROVENANCE_MARKS:
        raise ValueError(f"{mark_field} must be one of {', '.join(PROVENANCE_MARKS)}")


def check_referent(
    mark: str | None,
    *,
    file: str | None,
    revision: str | None,
    file_field: str = "file",
    revision_field: str = "revision",
) -> None:
    """Enforce that a mark names the field its meaning depends on.

    `MEASURED` asserts only about the bytes the digest already names,
    so it requires neither field. `RECOVERED` asserts that a file
    survived, and `RE_DERIVED` asserts a revision the bytes came
    from. Each names its referent, or no consumer can check the mark.
    An uncheckable mark is unrepresentable, not merely discouraged.

    Args:
        mark: The recorded provenance mark, or None.
        file: The record's file referent, or None.
        revision: The record's revision referent, or None.
        file_field: The file field's name, for the message.
        revision_field: The revision field's name, for the message.

    Raises:
        ValueError: If `mark` is `RECOVERED` with no ``file``, or
            `RE_DERIVED` with no ``revision``.

    Examples:
        ```python
        from vramfit.domain.provenance import check_referent

        check_referent("recovered", file="calibration.txt", revision=None)
        ```
    """
    if mark == RECOVERED and file is None:
        raise ValueError(
            f'provenance "{RECOVERED}" requires {file_field} — the mark '
            "says the run's own file survived, so the record must name it"
        )
    if mark == RE_DERIVED and revision is None:
        raise ValueError(
            f'provenance "{RE_DERIVED}" requires {revision_field} — the '
            "mark says these are the pinned revision's bytes, so the "
            "record must name the revision"
        )
