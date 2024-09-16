"""Generate subtitle from saved transcription."""

from pathlib import Path

import click
from loguru import logger
from subtitle_alchemy import forge
from subtitle_alchemy import merge
from subtitle_alchemy import serde


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=True))
@click.argument("folder", type=click.Path(exists=False, file_okay=False))
@click.option(
    "--form",
    help="Subtitle format to generate",
    default="srt",
    show_default=True,
    type=str,
)
@click.option(
    "--threshold",
    help="Gap threshold in milliseconds",
    default=500,
    show_default=True,
    type=click.INT,
)
def generate(
    src: str,
    folder: str,
    form: str = "srt",
    threshold: int = 500,
) -> None:
    """Generate subtitle from saved transcription."""
    src, folder = Path(src), Path(folder)
    key, txt, tl = serde.load(src)
    gap = merge.tl2gap(tl, th=threshold)
    instr = merge.gap2instr(gap)
    txt_ = merge.merge_txt(txt, instr)
    tl_ = merge.merge_tl(tl, instr)
    if form == "srt":
        forge.srt(tl_, txt_, folder / f"{key}.srt")
    else:
        logger.error(f"Unsupported format: {form}")
        raise ValueError()
