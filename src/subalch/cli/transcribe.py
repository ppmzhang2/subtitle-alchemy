"""STT command line interface."""

from pathlib import Path

import click
from subalch.stt import transcribe


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=True))
@click.argument("dst_dir", type=click.Path(exists=False, file_okay=False))
@click.option(
    "--model",
    help="STT model, default paraformer-zh",
    default="paraformer-zh",
    show_default=True,
    type=str,
)
@click.option(
    "--hotword",
    help="Hot words to detect, separated by space",
    default="",
    show_default=True,
    type=str,
)
def transcribe_cli(
    src: Path,
    dst_dir: Path,
    model: str,
    hotword: str = "",
) -> None:
    """Transcribe audio files in a directory."""
    res = transcribe(src, model, hotword)
    with open(dst_dir, "w") as f:
        f.write(res)
