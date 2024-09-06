"""All CLI commands here."""

import click
from subalch.cli.generate import generate
from subalch.cli.transcribe import transcribe


@click.group()
def cli() -> None:
    """Subalch CLI."""


cli.add_command(transcribe)
cli.add_command(generate)
