"""Ports: the protocols the domain's callers and collaborators satisfy.

Ports are structural (`typing.Protocol`) — adapters implement them by
shape, not inheritance. Only outbound (driven) ports exist today; the
domain's inbound API is its plain public functions.

Attributes:
    outbound: Driven-side protocols (`SensitivityMapSource`,
        `RecipeSink`, `ModelShapeSource`).

Examples:
    Type an orchestration step against a port, not an adapter:

    ```python
    from vramfit.ports.outbound import SensitivityMapSource


    def load(source: SensitivityMapSource):
        return source.load()
    ```

See Also:
    - [vramfit.adapters][]: The implementations.
"""

from __future__ import annotations
