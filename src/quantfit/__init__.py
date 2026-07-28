"""Selective per-layer quantization to fit large open models on a single GPU.

quantfit measures per-layer quantization sensitivity, solves for a
mixed-precision recipe under a VRAM budget, and packs the result for a
target runtime.

Attributes:
    __version__ (str): The installed package version.

Examples:
    Check the installed version:

    ```python
    import quantfit

    print(quantfit.__version__)
    ```

See Also:
    - [quantfit.adapters.inbound.cli][]: The ``quantfit`` console script
      entry point.
"""

from __future__ import annotations

__version__ = "0.1.0"
