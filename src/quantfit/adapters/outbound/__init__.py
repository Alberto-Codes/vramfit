"""Outbound (driven) adapters: implementations of the outbound ports.

JSON artifact files (sensitivity map, recipe) and Hugging Face model
configs. Serialization, schema versioning (``quantfit_schema``), and
validation are owned here — domain objects never see JSON.

Attributes:
    json_common: Shared validation helpers and the schema envelope.
    sensitivity_map_json: Sensitivity-map file IO.
    recipe_json: Recipe file IO.
    hf_config: Hugging Face ``config.json`` parsing.

Examples:
    Round-trip a recipe through files:

    ```python
    from quantfit.adapters.outbound.recipe_json import (
        load_recipe,
        save_recipe,
    )

    save_recipe(recipe, path)
    assert load_recipe(path) == recipe
    ```

See Also:
    - [quantfit.ports.outbound][]: The protocols implemented here.
"""

from __future__ import annotations
