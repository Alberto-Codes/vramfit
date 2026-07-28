from __future__ import annotations

import os

from hypothesis import settings

# Two hypothesis profiles per ADR-0009: "fast" keeps pre-commit quick,
# "thorough" runs on pre-push and CI via HYPOTHESIS_PROFILE=thorough.
settings.register_profile("fast", max_examples=25, deadline=None)
settings.register_profile("thorough", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "fast"))
