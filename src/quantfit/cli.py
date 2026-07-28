"""Typer CLI entry point for quantfit.

Exposes the ``quantfit`` console script with the scan/plan/pack pipeline
commands. Only ``version`` is implemented in the initial scaffold; the
pipeline commands are stubs that exit with a clear message.

Examples:
    Show the installed version:

    ```console
    $ quantfit version
    quantfit 0.1.0
    ```

See Also:
    - [quantfit][]: Package root exposing ``__version__``.
"""

from __future__ import annotations

import typer

from quantfit import __version__

app = typer.Typer(
    name="quantfit",
    help="Selective per-layer quantization to fit large models on a single GPU.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the installed quantfit version.

    Examples:
        Command line usage:

        ```console
        $ quantfit version
        quantfit 0.1.0
        ```
    """
    typer.echo(f"quantfit {__version__}")


@app.command()
def scan() -> None:
    """Measure per-layer quantization sensitivity (not yet implemented).

    Raises:
        typer.Exit: Always, with exit code 1, until the scan pipeline lands.

    Examples:
        Command line usage:

        ```console
        $ quantfit scan
        scan is not implemented yet -- see the roadmap in the README.
        ```
    """
    typer.echo("scan is not implemented yet -- see the roadmap in the README.")
    raise typer.Exit(code=1)
