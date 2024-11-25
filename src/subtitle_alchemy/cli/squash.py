"""Squash SRT subtitle files."""

from pathlib import Path

import click
from loguru import logger
from subtitle_alchemy import merge
from subtitle_alchemy.forger import gen_srt
from subtitle_alchemy.parser import srt


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
    default=300000,
    show_default=True,
    type=click.INT,
)
def squash(
    src: str,
    folder: str,
    form: str = "srt",
    threshold: int = 500,
) -> None:
    """Squash SRT subtitle files."""
    with open(src) as f:
        txt, tl = srt.generate(f.read())

    adh = merge.tl2adh(tl, th=threshold)
    instr = merge.adh2instr(adh)
    txt_ = merge.merge_txt(txt, instr)
    tl_ = merge.merge_tl(tl, instr)
    if form == "srt":
        gen_srt(tl_, txt_, Path(folder) / f"{Path(src).stem}.srt")
    else:
        logger.error(f"Unsupported format: {form}")
        raise ValueError()
