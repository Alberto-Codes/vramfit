"""The provenance mark's two rules, shared by every marked digest.

One vocabulary serves the evals sidecar's corpus reference (#589) and
the sensitivity map's calibration file (issue #558). These tests hold
the rules at their one home, so a third artifact inherits them
instead of restating them.
"""

from __future__ import annotations

import pytest

from vramfit.domain.provenance import (
    MEASURED,
    PROVENANCE_MARKS,
    RE_DERIVED,
    RECOVERED,
    check_mark_pairs_with_digest,
    check_referent,
)

pytestmark = pytest.mark.unit

DIGEST = "a" * 64


class TestTheVocabulary:
    def test_the_three_marks_are_the_whole_vocabulary(self) -> None:
        # A fourth value would be a second vocabulary. The sidecar
        # reads this tuple, not a copy of it.
        assert PROVENANCE_MARKS == (MEASURED, RECOVERED, RE_DERIVED)


class TestDigestAndMarkPair:
    def test_a_digest_with_its_mark_is_accepted(self) -> None:
        check_mark_pairs_with_digest(
            DIGEST, MEASURED, digest_field="sha256", mark_field="provenance"
        )

    def test_neither_recorded_is_accepted(self) -> None:
        # NOT RECORDED is the honest record, and it needs no mark.
        check_mark_pairs_with_digest(
            None, None, digest_field="sha256", mark_field="provenance"
        )

    def test_a_digest_without_a_mark_raises(self) -> None:
        with pytest.raises(ValueError, match="must pair"):
            check_mark_pairs_with_digest(
                DIGEST, None, digest_field="sha256", mark_field="provenance"
            )

    def test_a_mark_without_a_digest_raises(self) -> None:
        with pytest.raises(ValueError, match="must pair"):
            check_mark_pairs_with_digest(
                None, MEASURED, digest_field="sha256", mark_field="provenance"
            )

    def test_an_unknown_mark_raises(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            check_mark_pairs_with_digest(
                DIGEST, "guessed", digest_field="sha256", mark_field="provenance"
            )

    def test_the_message_names_the_caller_s_own_field_spellings(self) -> None:
        # One rule, two schemas. The map spells the digest
        # `calibration_sha256`, so its reader must say that.
        with pytest.raises(ValueError, match="calibration_sha256"):
            check_mark_pairs_with_digest(
                DIGEST,
                None,
                digest_field="calibration_sha256",
                mark_field="calibration_provenance",
            )


class TestAMarkNamesItsReferent:
    def test_measured_needs_no_referent(self) -> None:
        # It asserts only about the bytes the digest already names.
        check_referent(MEASURED, file=None, revision=None)

    def test_recovered_without_a_file_raises(self) -> None:
        with pytest.raises(ValueError, match="requires file"):
            check_referent(RECOVERED, file=None, revision="b08601e")

    def test_re_derived_without_a_revision_raises(self) -> None:
        with pytest.raises(ValueError, match="requires revision"):
            check_referent(RE_DERIVED, file="calibration.txt", revision=None)

    def test_recovered_with_its_file_is_accepted(self) -> None:
        check_referent(RECOVERED, file="calibration.txt", revision=None)

    def test_re_derived_with_its_revision_is_accepted(self) -> None:
        check_referent(RE_DERIVED, file=None, revision="b08601e")

    def test_no_mark_needs_no_referent(self) -> None:
        check_referent(None, file=None, revision=None)

    def test_the_referent_message_names_the_caller_s_field(self) -> None:
        with pytest.raises(ValueError, match="calibration_revision"):
            check_referent(
                RE_DERIVED,
                file=None,
                revision=None,
                revision_field="calibration_revision",
            )
