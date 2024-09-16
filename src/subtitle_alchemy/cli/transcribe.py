"""Transcribe audio and save transcription."""

from pathlib import Path

import click
from subtitle_alchemy import serde
from subtitle_alchemy import stt


@click.command()
@click.argument("audio", type=click.Path(exists=True, file_okay=True))
@click.argument("folder", type=click.Path(exists=False, file_okay=False))
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
def transcribe(
    audio: str,
    folder: str,
    model: str = "paraformer-zh",
    hotword: str = "",
) -> None:
    """Transcribe audio and save transcription."""
    audio, folder = Path(audio), Path(folder)
    key, txt, tl = stt.generate(model, audio, hotword)
    serde.save(key, folder, txt, tl)
