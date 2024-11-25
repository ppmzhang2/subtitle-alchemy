"""Align ground truth and prediction."""

from pathlib import Path

import click
from loguru import logger
from subtitle_alchemy import merge
from subtitle_alchemy.align import get_aligned_index
from subtitle_alchemy.align import get_aligned_tl
from subtitle_alchemy.forger import gen_srt
from subtitle_alchemy.utils import punc
from subtitle_alchemy.utils import sl


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
    p_pred, p_truth, p_folder = Path(pred), Path(truth), Path(folder)
    key, txt_pred, tl_pred, _ = sl.load(p_pred)  # TODO: support punctuation

    with open(p_truth) as f:
        speech = f.read().strip()
    # TODO: support punctuation
    txt_aligned, _ = punc.separate(speech)
    idx_aligned = get_aligned_index(txt_aligned, txt_pred)
    tl_aligned = get_aligned_tl(idx_aligned, tl_pred)

    gap = merge.tl2adh(tl_aligned, th=threshold)
    instr = merge.adh2instr(gap)
    txt_ = merge.merge_txt(txt_aligned, instr)
    tl_ = merge.merge_tl(tl_aligned, instr)
    if form == "srt":
        gen_srt(tl_, txt_, p_folder / f"{key}.srt")
    else:
        logger.error(f"Unsupported format: {form}")
        raise ValueError()
