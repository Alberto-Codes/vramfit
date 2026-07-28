from __future__ import annotations

import pytest
from typer.testing import CliRunner

from quantfit import __version__
from quantfit.cli import app

runner = CliRunner()


@pytest.mark.unit
def test_version_command_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.output == f"quantfit {__version__}\n"


@pytest.mark.unit
def test_scan_command_unimplemented_exits_nonzero() -> None:
    result = runner.invoke(app, ["scan"])

    assert result.exit_code == 1
    assert "not implemented" in result.output


@pytest.mark.unit
def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "scan" in result.output
    assert "version" in result.output
