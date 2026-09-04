"""Hermetic tests for ``scripts/publish_card.py``.

No test touches the network. A fake Hub client stands in for
``huggingface_hub.HfApi`` and records what the script uploaded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_script() -> ModuleType:
    """Import the script by path — ``scripts/`` is not an installed package."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "publish_card.py"
    spec = importlib.util.spec_from_file_location("publish_card", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


publish = _load_script()

pytestmark = pytest.mark.unit

MODEL_CARD = """---
license: other
quantized_by: Alberto-Codes
tags:
  - vramfit
---

<!-- Upload this file verbatim. -->

# Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF

Body text.
"""

DATASET_CARD = """---
license: cc-by-4.0
tags:
  - vramfit
---

# gemma-4-31B-it-sensitivity-maps

Body text.
"""


class FakeHub:
    """Records uploads and serves them back, optionally corrupted."""

    def __init__(self, *, corrupt: str | None = None) -> None:
        """``corrupt`` names one path whose stored copy gains a trailing byte."""
        self.uploads: list[tuple[str, str, str]] = []
        self.stored: dict[str, bytes] = {}
        self.corrupt = corrupt
        self.revisions: list[str | None] = []

    def upload_file(
        self,
        *,
        path_or_fileobj: str,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> SimpleNamespace:
        self.uploads.append((repo_id, repo_type, path_in_repo))
        data = Path(path_or_fileobj).read_bytes()
        if path_in_repo == self.corrupt:
            data += b"\n"
        self.stored[path_in_repo] = data
        return SimpleNamespace(oid=f"sha-{len(self.uploads)}")

    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str,
        revision: str | None,
        cache_dir: str,
        force_download: bool,
    ) -> str:
        self.revisions.append(revision)
        target = Path(cache_dir) / filename
        target.write_bytes(self.stored[filename])
        return str(target)


def _publication(tmp_path: Path, card: str = MODEL_CARD) -> Path:
    directory = tmp_path / "pub"
    directory.mkdir()
    (directory / "README.md").write_text(card, encoding="utf-8")
    return directory


def test_resolve_repo_model_card_uses_quantized_by_and_title() -> None:
    assert (
        publish.resolve_repo(MODEL_CARD)
        == "Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF"
    )


def test_resolve_repo_without_title_raises() -> None:
    with pytest.raises(publish.UploadError, match="H1 title"):
        publish.resolve_repo("no heading here\n")


def test_resolve_repo_without_quantized_by_raises() -> None:
    with pytest.raises(publish.UploadError, match="quantized_by"):
        publish.resolve_repo(DATASET_CARD)


def test_card_dry_run_prints_target_and_hash_without_uploading(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _publication(tmp_path)
    hub = FakeHub()

    status = publish.run(["card", str(directory), "--dry-run"], api_factory=lambda: hub)

    out = capsys.readouterr().out
    expected = hashlib.sha256(MODEL_CARD.encode()).hexdigest()
    assert status == 0
    assert (
        "model repo Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF"
        in out
    )
    assert expected in out
    assert "dry run" in out
    assert hub.uploads == []


def test_card_upload_verifies_bytes_and_prints_ledger_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _publication(tmp_path)
    hub = FakeHub()

    status = publish.run(["card", str(directory)], api_factory=lambda: hub)

    out = capsys.readouterr().out
    expected = hashlib.sha256(MODEL_CARD.encode()).hexdigest()
    assert status == 0
    assert hub.uploads == [
        (
            "Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF",
            "model",
            "README.md",
        )
    ]
    assert hub.revisions == ["sha-1"]
    assert (
        f"ledger: | `README.md` | `{expected}` | {len(MODEL_CARD.encode()):,} |\n"
        in out
    )


def test_card_upload_mismatch_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _publication(tmp_path)
    hub = FakeHub(corrupt="README.md")

    status = publish.run(["card", str(directory)], api_factory=lambda: hub)

    err = capsys.readouterr().err
    assert status == 1
    assert "published bytes differ" in err


def test_card_missing_readme_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "empty"
    directory.mkdir()

    status = publish.run(["card", str(directory)], api_factory=FakeHub)

    assert status == 1
    assert "holds no README.md" in capsys.readouterr().err


def test_card_rejects_repo_flag(tmp_path: Path) -> None:
    directory = _publication(tmp_path)

    with pytest.raises(SystemExit):
        publish.run(["card", str(directory), "--repo", "o/n"], api_factory=FakeHub)


def test_missing_directory_exits_two(tmp_path: Path) -> None:
    assert publish.run(["card", str(tmp_path / "nope")], api_factory=FakeHub) == 2


def test_dataset_uploads_files_then_card_and_prints_every_ledger_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _publication(tmp_path, DATASET_CARD)
    (directory / "sensitivity-32k.json").write_bytes(b'{"vramfit_schema": 3}')
    (directory / "imatrix.gguf").write_bytes(b"GGUF\x00\x01")
    (directory / ".hidden").write_bytes(b"skip")
    hub = FakeHub()

    status = publish.run(
        [
            "dataset",
            str(directory),
            "--repo",
            "Alberto-Codes/gemma-4-31B-it-sensitivity-maps",
        ],
        api_factory=lambda: hub,
    )

    out = capsys.readouterr().out
    assert status == 0
    assert [name for _, _, name in hub.uploads] == [
        "imatrix.gguf",
        "sensitivity-32k.json",
        "README.md",
    ]
    assert {kind for _, kind, _ in hub.uploads} == {"dataset"}
    assert {repo for repo, _, _ in hub.uploads} == {
        "Alberto-Codes/gemma-4-31B-it-sensitivity-maps"
    }
    for name in ("imatrix.gguf", "sensitivity-32k.json", "README.md"):
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert f"ledger: | `{name}` | `{digest}` |\n" in out


def test_dataset_without_repo_flag_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "maps"
    directory.mkdir()
    (directory / "map.json").write_bytes(b"{}")

    with pytest.raises(SystemExit) as exc:
        publish.run(["dataset", str(directory)], api_factory=FakeHub)

    assert exc.value.code == 2
    assert "--repo" in capsys.readouterr().err


def test_dataset_empty_directory_exits_nonzero_without_uploading(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "maps"
    directory.mkdir()
    (directory / ".hidden").write_bytes(b"skip")
    (directory / "nested").mkdir()
    hub = FakeHub()

    status = publish.run(
        ["dataset", str(directory), "--repo", "someone/maps"], api_factory=lambda: hub
    )

    assert status == 1
    assert "holds no files to upload" in capsys.readouterr().err
    assert hub.uploads == []


def test_dataset_without_card_uploads_the_files(tmp_path: Path) -> None:
    directory = tmp_path / "maps"
    directory.mkdir()
    (directory / "map.json").write_bytes(b"{}")
    hub = FakeHub()

    status = publish.run(
        ["dataset", str(directory), "--repo", "someone/maps"], api_factory=lambda: hub
    )

    assert status == 0
    assert hub.uploads == [("someone/maps", "dataset", "map.json")]


def test_dataset_mismatch_on_one_file_exits_nonzero(tmp_path: Path) -> None:
    directory = _publication(tmp_path, DATASET_CARD)
    (directory / "map.json").write_bytes(b"{}")
    hub = FakeHub(corrupt="map.json")

    status = publish.run(
        ["dataset", str(directory), "--repo", "someone/maps"], api_factory=lambda: hub
    )

    assert status == 1


def test_hub_api_without_huggingface_hub_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(publish.UploadError, match="huggingface_hub is not installed"):
        publish._hub_api()


def test_real_card_dry_run_names_the_published_repo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path(__file__).resolve().parents[2]
    status = publish.run(
        ["card", str(root / "publication" / "model-card"), "--dry-run"],
        api_factory=FakeHub,
    )
    assert status == 0
    assert (
        "Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF"
        in capsys.readouterr().out
    )
