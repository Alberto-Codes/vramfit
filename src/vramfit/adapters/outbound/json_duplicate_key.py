"""Every JSON reader in `src/` and `scripts/` refuses a repeated key.

`json.loads` keeps the last value when a document repeats a key, and it
reports nothing (#262). RFC 8259 permits the repeat and states that the
behavior of receiving software is unpredictable. Readers disagree in
practice — Python and Go keep the last value, and rapidjson keeps the
first. So a document that repeats a key means two things at once, and
vramfit refuses it.

Four artifact readers apply the rule through `_load_json`. Three readers
outside it apply the rule directly: the run log, a Hugging Face
``config.json``, and a safetensors shard index (#283).

`scripts/backfill_tensor_sizes.py` applies the rule directly too, at
both of its reads: the sensitivity map it annotates, and the header
inside each ``.safetensors`` shard (#286). The script stays outside the
package, so this list crosses that boundary.

The hook lives here rather than in `json_common`, because those three
readers share nothing else with the artifact readers. `json_common`
carries `ArtifactError`, the ``vramfit_schema`` envelope, and the
path-reporting extractors. A run log and a publisher's config file have
none of those. A shared module keeps every import public.

Examples:
    Refuse an object that repeats a key:

    ```python
    from vramfit.adapters.outbound.json_duplicate_key import (
        DuplicateKeyError,
        object_from_pairs,
    )

    try:
        object_from_pairs([("ppl", 999.0), ("ppl", 8.5543)])
    except DuplicateKeyError as exc:
        print(exc.key)
    ```

See Also:
    - [vramfit.adapters.outbound.json_common][]: The artifact readers.
    - [vramfit.adapters.outbound.run_log_jsonl][]: The run-log reader.
    - [vramfit.adapters.outbound.hf_config][]: The publisher's
      ``config.json`` reader.
    - [vramfit.adapters.outbound.scan.offload][]: The shard-index reader.
"""

from __future__ import annotations

from typing import Any

from vramfit.domain.errors import VramfitError


class DuplicateKeyError(VramfitError):
    """One JSON object defined the same key twice.

    `object_from_pairs` raises this. Each reader prefixes `message` with
    the file it was reading. A reader with callers converts the error to
    the type those callers already catch. A command-line entry point
    prints the message and exits 1.

    The class sits under the `VramfitError` root per ADR-0011 decision 5.
    It does not subclass `ValueError`. A catch-all `ValueError` clause
    therefore cannot relabel a structural refusal as a parse failure,
    whatever order the clauses take (#262).

    Attributes:
        key (str): The key the object defined twice.
        message (str): The refusal, carrying no file or line locator.
            The reader adds the locator it knows.

    Examples:
        Read the duplicated key off the refusal:

        ```python
        try:
            object_from_pairs([("bits", 4), ("bits", 3)])
        except DuplicateKeyError as exc:
            assert exc.key == "bits"
        ```
    """

    def __init__(self, key: str) -> None:
        """Record the duplicated key and build the refusal message.

        Args:
            key: The key the object defined twice.
        """
        self.key = key
        self.message = (
            f'duplicate key "{key}" — one JSON object defines it twice. '
            "The reader cannot choose between the two values. Delete one."
        )
        super().__init__(self.message)


def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object from its pairs, refusing a repeated key.

    Pass this as ``object_pairs_hook`` to `json.loads` or `json.load`.

    The hook sees one object's pairs and no ancestry, so it reports the
    key alone. A caller that knows the file adds that locator. Neither
    can build the ``$.tier1.ppl`` path the artifact extractors report.

    The check reads the object under construction, so the hook walks the
    pairs once. Every JSON object in every document passes through here.

    Args:
        pairs: The object's key-value pairs, in document order.

    Returns:
        The pairs as a dict.

    Raises:
        DuplicateKeyError: If a key appears more than once in the object.
    """
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise DuplicateKeyError(key)
        obj[key] = value
    return obj
