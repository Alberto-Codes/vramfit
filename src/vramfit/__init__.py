"""Selective per-layer quantization to fit large open models on a single GPU.

vramfit measures per-layer quantization sensitivity, solves for a
mixed-precision recipe under a VRAM budget, and packs the result for a
target runtime.

Attributes:
    __version__ (str): The installed package version.

Examples:
    Check the installed version:

    ```python
    import vramfit

    print(vramfit.__version__)
    ```

See Also:
    - [vramfit.adapters.inbound.cli][]: The ``vramfit`` console script
      entry point.
"""

from __future__ import annotations

__version__ = "0.1.0"
