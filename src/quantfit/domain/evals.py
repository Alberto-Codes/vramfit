"""Evaluation results as domain values: the evals sidecar (ADR-0025).

One `EvalsSidecar` records one evaluated artifact's scoreboard
evidence: the three tiers with their settings, and the toolchain that
produced the numbers. Every numeric field records what the instrument
printed — the sidecar exists so the model card never transcribes.
Each tier block is optional, and at least one must be present: the
i-quant baselines carry tiers 1-2 only, the certified pair carries
all three. Invariants live in ``__post_init__`` (finite numbers,
non-negative standard errors, a well-formed SHA-256). Serialization
belongs to the JSON adapter (ADR-0008), never here.

Examples:
    Build a tier-1-only sidecar:

    ```python
    sidecar = EvalsSidecar(
        artifact=EvaluatedArtifact("m.gguf", "0" * 64, 1024),
        toolchain=EvalToolchain(llama_cpp_build="b10172"),
        tier1=Tier1Result("2026-08-10", "wikitext-2-test", 564, 8.52, 0.063),
    )
    ```

See Also:
    - [quantfit.ports.outbound][]: `EvalsSidecarSink`, which carries
      these values out of the application.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_HEX_DIGITS = frozenset("0123456789abcdef")
_SHA256_HEX_LEN = 64
_PERCENT_MAX = 100


def _check_finite(value: float, name: str) -> None:
    """Raise unless ``value`` is a finite number.

    Args:
        value: The number to check.
        name: Field name for the error message.

    Raises:
        ValueError: If ``value`` is NaN or infinite.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _check_stderr(value: float, name: str) -> None:
    """Raise unless ``value`` is a finite, non-negative standard error.

    Args:
        value: The standard error to check.
        name: Field name for the error message.

    Raises:
        ValueError: If ``value`` is NaN, infinite, or negative.
    """
    _check_finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must not be negative")


@dataclass(frozen=True, slots=True)
class EvaluatedArtifact:
    """The exact file the sidecar's numbers describe.

    Attributes:
        file (str): The artifact's file name, e.g. ``model.gguf``.
        sha256 (str): SHA-256 of the file, 64 lowercase hex digits.
        size_bytes (int): The file's size in bytes.

    Examples:
        Name a packed model:

        ```python
        EvaluatedArtifact("model.gguf", "0" * 64, 21_860_214_272)
        ```
    """

    file: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        """Enforce the file-identity invariants.

        Raises:
            ValueError: If ``file`` is empty, ``sha256`` is not 64
                lowercase hex digits, or ``size_bytes`` is not
                positive.
        """
        if not self.file:
            raise ValueError("file must not be empty")
        if len(self.sha256) != _SHA256_HEX_LEN or not set(self.sha256) <= _HEX_DIGITS:
            raise ValueError("sha256 must be 64 lowercase hex digits")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")


@dataclass(frozen=True, slots=True)
class EvalToolchain:
    """The instruments that produced the sidecar's numbers.

    The tier-3 fields default to None: an artifact evaluated at
    tiers 1-2 only names no harness. `EvalsSidecar` enforces the
    pairing — a present tier 3 requires all three.

    Attributes:
        llama_cpp_build (str): The llama.cpp build behind every tier.
        lm_eval (str | None): lm-evaluation-harness version, tier 3 only.
        llama_cpp_python (str | None): Binding version, tier 3 only.
        lane (str | None): The recorded harness lane (ADR-0024), tier 3
            only.

    Examples:
        A tiers-1-2-only toolchain names the build alone:

        ```python
        EvalToolchain(llama_cpp_build="b10172")
        ```
    """

    llama_cpp_build: str
    lm_eval: str | None = None
    llama_cpp_python: str | None = None
    lane: str | None = None

    def __post_init__(self) -> None:
        """Enforce the instrument-naming invariants.

        Raises:
            ValueError: If ``llama_cpp_build`` is empty, or an
                optional field is the empty string instead of None.
        """
        if not self.llama_cpp_build:
            raise ValueError("llama_cpp_build must not be empty")
        for name in ("lm_eval", "llama_cpp_python", "lane"):
            value = getattr(self, name)
            if value == "":
                raise ValueError(f"{name} must not be empty — use None")


@dataclass(frozen=True, slots=True)
class Tier1Result:
    """Tier 1: perplexity over the full held-out set.

    Attributes:
        date (str): Run date, ``YYYY-MM-DD``.
        dataset (str): The held-out text, e.g. ``wikitext-2-test``.
        chunks (int): Chunk count the estimate ran over.
        ppl (float): Final perplexity estimate.
        ppl_stderr (float): Its standard error.

    Examples:
        Record a full-set estimate:

        ```python
        Tier1Result("2026-08-10", "wikitext-2-test", 564, 8.52, 0.063)
        ```
    """

    date: str
    dataset: str
    chunks: int
    ppl: float
    ppl_stderr: float

    def __post_init__(self) -> None:
        """Enforce the tier-1 invariants.

        Raises:
            ValueError: If a string field is empty, ``chunks`` or
                ``ppl`` is not positive, or a number is not finite.
        """
        if not self.date:
            raise ValueError("date must not be empty")
        if not self.dataset:
            raise ValueError("dataset must not be empty")
        if self.chunks <= 0:
            raise ValueError("chunks must be positive")
        _check_finite(self.ppl, "ppl")
        if self.ppl <= 0:
            raise ValueError("ppl must be positive")
        _check_stderr(self.ppl_stderr, "ppl_stderr")


@dataclass(frozen=True, slots=True)
class Tier2Window:
    """One KL-divergence window against the reference logits.

    Attributes:
        date (str): Run date, ``YYYY-MM-DD``.
        chunks (int): Window size in chunks.
        mean_kld (float): Mean KL divergence over the window.
        kld_stderr (float): Its standard error.
        same_top_pct (float): Same-top-token rate, percent.
        same_top_stderr_pct (float): Its standard error, percent.

    Examples:
        Record the full window:

        ```python
        Tier2Window("2026-08-10", 564, 0.287, 0.0032, 82.917, 0.099)
        ```
    """

    date: str
    chunks: int
    mean_kld: float
    kld_stderr: float
    same_top_pct: float
    same_top_stderr_pct: float

    def __post_init__(self) -> None:
        """Enforce the window invariants.

        Raises:
            ValueError: If ``date`` is empty, ``chunks`` is not
                positive, ``mean_kld`` is negative, ``same_top_pct``
                is outside 0-100, or a number is not finite.
        """
        if not self.date:
            raise ValueError("date must not be empty")
        if self.chunks <= 0:
            raise ValueError("chunks must be positive")
        _check_finite(self.mean_kld, "mean_kld")
        if self.mean_kld < 0:
            raise ValueError("mean_kld must not be negative")
        _check_stderr(self.kld_stderr, "kld_stderr")
        _check_finite(self.same_top_pct, "same_top_pct")
        if not 0 <= self.same_top_pct <= _PERCENT_MAX:
            raise ValueError("same_top_pct must be between 0 and 100")
        _check_stderr(self.same_top_stderr_pct, "same_top_stderr_pct")


@dataclass(frozen=True, slots=True)
class Tier2Result:
    """Tier 2: KL divergence against a reference, per window.

    Each window carries its own run date — one artifact's windows
    run on different days.

    Attributes:
        reference (str): The reference logits' precision, e.g. ``f16``.
        dataset (str): The held-out text the windows run over.
        windows (tuple[Tier2Window, ...]): Measured windows, unique
            chunk counts.

    Examples:
        Record one window against the f16 reference:

        ```python
        Tier2Result("f16", "wikitext-2-test", (window,))
        ```
    """

    reference: str
    dataset: str
    windows: tuple[Tier2Window, ...]

    def __post_init__(self) -> None:
        """Enforce the tier-2 invariants.

        Raises:
            ValueError: If a string field is empty, ``windows`` is
                empty, or two windows share a chunk count.
        """
        if not self.reference:
            raise ValueError("reference must not be empty")
        if not self.dataset:
            raise ValueError("dataset must not be empty")
        if not self.windows:
            raise ValueError("windows must not be empty")
        if len({w.chunks for w in self.windows}) != len(self.windows):
            raise ValueError("window chunk counts must be unique")


@dataclass(frozen=True, slots=True)
class Tier3Task:
    """One lm-evaluation-harness task's aggregate result.

    Attributes:
        date (str): Run date, ``YYYY-MM-DD``.
        name (str): Task name, e.g. ``mmlu``.
        version (str): The harness's per-task version (ADR-0025).
        few_shot (int): Few-shot setting (ADR-0024).
        n (int): Evaluated item count — the full split, never a sample.
        metric (str): The reported metric, e.g. ``acc_norm`` or
            ``exact_match,strict-match``.
        score (float): The metric's value.
        stderr (float): Its standard error.
        wall_clock_seconds (float): Measured task duration.

    Examples:
        Record a Winogrande row:

        ```python
        Tier3Task(
            "2026-08-10", "winogrande", "1.0", 5, 1267, "acc", 0.784, 0.012, 351.6
        )
        ```
    """

    date: str
    name: str
    version: str
    few_shot: int
    n: int
    metric: str
    score: float
    stderr: float
    wall_clock_seconds: float

    def __post_init__(self) -> None:
        """Enforce the task-row invariants.

        Raises:
            ValueError: If a string field is empty, ``few_shot`` is
                negative, ``n`` or ``wall_clock_seconds`` is not
                positive, or a number is not finite.
        """
        if not self.date:
            raise ValueError("date must not be empty")
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.version:
            raise ValueError("version must not be empty")
        if self.few_shot < 0:
            raise ValueError("few_shot must not be negative")
        if self.n <= 0:
            raise ValueError("n must be positive")
        if not self.metric:
            raise ValueError("metric must not be empty")
        _check_finite(self.score, "score")
        _check_stderr(self.stderr, "stderr")
        _check_finite(self.wall_clock_seconds, "wall_clock_seconds")
        if self.wall_clock_seconds <= 0:
            raise ValueError("wall_clock_seconds must be positive")


@dataclass(frozen=True, slots=True)
class Tier3Result:
    """Tier 3: the fixed task slice's aggregates (ADR-0024).

    Each task carries its own run date — one slice can cross
    midnight.

    Attributes:
        tasks (tuple[Tier3Task, ...]): Per-task aggregates, unique
            names.

    Examples:
        Record a one-task slice:

        ```python
        Tier3Result((task,))
        ```
    """

    tasks: tuple[Tier3Task, ...]

    def __post_init__(self) -> None:
        """Enforce the tier-3 invariants.

        Raises:
            ValueError: If ``tasks`` is empty, or two tasks share a
                name.
        """
        if not self.tasks:
            raise ValueError("tasks must not be empty")
        if len({t.name for t in self.tasks}) != len(self.tasks):
            raise ValueError("task names must be unique")


@dataclass(frozen=True, slots=True)
class EvalsSidecar:
    """One evaluated artifact's complete scoreboard evidence.

    Attributes:
        artifact (EvaluatedArtifact): The file the numbers describe.
        toolchain (EvalToolchain): The instruments behind the numbers.
        tier1 (Tier1Result | None): Perplexity, when measured.
        tier2 (Tier2Result | None): KL divergence, when measured.
        tier3 (Tier3Result | None): The task slice, when measured.

    Examples:
        A tier-1-only baseline record:

        ```python
        EvalsSidecar(artifact=artifact, toolchain=toolchain, tier1=tier1)
        ```
    """

    artifact: EvaluatedArtifact
    toolchain: EvalToolchain
    tier1: Tier1Result | None = None
    tier2: Tier2Result | None = None
    tier3: Tier3Result | None = None

    def __post_init__(self) -> None:
        """Enforce the whole-record invariants.

        The harness toolchain fields pair with tier 3 in both
        directions (ADR-0025): a present tier 3 requires all three,
        and an absent tier 3 forbids them — no harness ran.

        Raises:
            ValueError: If every tier is absent, if tier 3 is present
                without the harness toolchain fields, or if a harness
                field is present without tier 3.
        """
        if self.tier1 is None and self.tier2 is None and self.tier3 is None:
            raise ValueError("at least one tier must be present")
        harness_absent = (
            self.toolchain.lm_eval is None
            or self.toolchain.llama_cpp_python is None
            or self.toolchain.lane is None
        )
        harness_present = (
            self.toolchain.lm_eval is not None
            or self.toolchain.llama_cpp_python is not None
            or self.toolchain.lane is not None
        )
        if self.tier3 is not None and harness_absent:
            raise ValueError(
                "tier3 requires the toolchain's lm_eval, llama_cpp_python, "
                "and lane — the sidecar names what produced the numbers "
                "(ADR-0025)"
            )
        if self.tier3 is None and harness_present:
            raise ValueError(
                "the toolchain's lm_eval, llama_cpp_python, and lane pair "
                "with tier3 — without the task slice no harness ran "
                "(ADR-0025)"
            )
