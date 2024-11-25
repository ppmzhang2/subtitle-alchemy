"""Transcribe audio and save transcription."""

from pathlib import Path

import click
from subtitle_alchemy.parser import stt
from subtitle_alchemy.utils import sl


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
    p_audio = Path(audio)
    p_folder = Path(folder)
    key, txt, tl, punc = stt.generate(model, p_audio, hotword)
    sl.save(key, p_folder, txt, tl, punc)
