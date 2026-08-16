"""The shared duplicate-key hook, tested directly (#283).

Four artifact readers reach the hook through `_load_json`, and three
readers outside it install the hook themselves. The behavior every one
of them depends on is pinned here once.
"""

from __future__ import annotations

import json

import pytest

from vramfit.adapters.outbound.json_duplicate_key import (
    DuplicateKeyError,
    object_from_pairs,
)
from vramfit.domain.errors import VramfitError

pytestmark = pytest.mark.unit


def test_unique_pairs_build_the_object() -> None:
    assert object_from_pairs([("a", 1), ("b", 2)]) == {"a": 1, "b": 2}


def test_repeated_key_raises_and_carries_the_key() -> None:
    with pytest.raises(DuplicateKeyError) as caught:
        object_from_pairs([("ppl", 999.0), ("ppl", 8.5543)])

    assert caught.value.key == "ppl"
    assert 'duplicate key "ppl"' in caught.value.message


def test_the_refusal_sits_under_the_error_root() -> None:
    # ADR-0011 decision 5 puts every vramfit exception under the root.
    assert issubclass(DuplicateKeyError, VramfitError)


def test_the_refusal_is_not_a_value_error() -> None:
    # A catch-all `except ValueError` clause must not relabel a
    # structural refusal as a parse failure (#262). Every reader that
    # installs the hook relies on this, because each one already carries
    # a `ValueError` clause of its own.
    assert not issubclass(DuplicateKeyError, ValueError)


def test_the_hook_reaches_a_nested_object() -> None:
    with pytest.raises(DuplicateKeyError, match="ppl"):
        json.loads(
            '{"tier1": {"ppl": 1.0, "ppl": 2.0}}', object_pairs_hook=object_from_pairs
        )


def test_the_hook_reaches_an_object_inside_a_list() -> None:
    with pytest.raises(DuplicateKeyError, match="name"):
        json.loads(
            '{"groups": [{"name": "a", "name": "b"}]}',
            object_pairs_hook=object_from_pairs,
        )


def test_the_same_key_in_sibling_objects_is_not_a_duplicate() -> None:
    parsed = json.loads(
        '{"a": {"bits": 4}, "b": {"bits": 3}}', object_pairs_hook=object_from_pairs
    )

    assert parsed == {"a": {"bits": 4}, "b": {"bits": 3}}
