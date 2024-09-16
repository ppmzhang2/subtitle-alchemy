"""All CLI commands here."""

import click
from subtitle_alchemy.cli.align import align
from subtitle_alchemy.cli.generate import generate
from subtitle_alchemy.cli.transcribe import transcribe


@click.group()
def cli() -> None:
    """Subalch CLI."""


cli.add_command(align)
cli.add_command(generate)
cli.add_command(transcribe)
