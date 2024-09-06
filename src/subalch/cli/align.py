"""Align ground truth and prediction."""

from pathlib import Path

import click
from loguru import logger
from subalch import build
from subalch import merge
from subalch.align import get_aligned_index
from subalch.align import get_aligned_tl
from subalch.serde import load
from subalch.utils import punc


@click.command()
@click.argument("pred", type=click.Path(exists=True, file_okay=True))
@click.argument("truth", type=click.Path(exists=False, file_okay=True))
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
def align(
    pred: str,
    truth: str,
    folder: str,
    form: str = "srt",
    threshold: int = 500,
) -> None:
    """Align predicted transcript with the ground truth speech."""
    pred, truth, folder = Path(pred), Path(truth), Path(folder)
    key, txt_pred, tl_pred = load(pred)

    with open(truth) as f:
        speech = f.read().strip()
    # TODO: support punctuation
    txt_aligned, _ = punc.separate(speech)
    idx_aligned = get_aligned_index(txt_aligned, txt_pred)
    tl_aligned = get_aligned_tl(idx_aligned, tl_pred)

    gap = merge.tl2gap(tl_aligned, th=threshold)
    instr = merge.gap2instr(gap)
    txt_ = merge.merge_txt(txt_aligned, instr)
    tl_ = merge.merge_tl(tl_aligned, instr)
    if form == "srt":
        build.srt(tl_, txt_, folder / f"{key}.srt")
    else:
        logger.error(f"Unsupported format: {form}")
        raise ValueError()
