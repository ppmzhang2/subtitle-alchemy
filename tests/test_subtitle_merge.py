"""Test merge module of subtitle package."""

import numpy as np
from subtitle_alchemy import merge

ARR_TL = np.array(
    [
        [0, 100],
        [300, 800],
        [1100, 1190],
        [1200, 1230],
        [1250, 1300],
        [1700, 1735],
        [1750, 1780],
        [1800, 1870],
        [1900, 1910],
        [2410, 2510],
        [2910, 2950],
        [3250, 3300],
    ],
    dtype=np.uint32,
)
ARR_TXT = np.array(
    ["哦", "对", "太", "对", "了", "以", "后", "处", "理", "啊", "好", "的"],
    dtype="U",
)
ARR_GAP = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
ARR_INSTR = np.array(
    [[0, 0], [1, 1], [2, 4], [5, 8], [9, 9], [10, 10], [11, 11]],
    dtype=np.uint32,
)
ARR_TL_MERGED = np.array(
    [
        [0, 100],
        [300, 800],
        [1100, 1300],
        [1700, 1910],
        [2410, 2510],
        [2910, 2950],
        [3250, 3300],
    ],
    dtype=np.uint32,
)
ARR_TXT_MERGED = np.array(
    ["哦", "对", "太对了", "以后处理", "啊", "好", "的"], dtype="U"
)


def test_subtitle_merge_tl2gap() -> None:
    """Test `merge.tl2gap` function."""
    gap = merge.tl2gap(ARR_TL, th=100)
    assert np.array_equal(gap, ARR_GAP)


def test_subtitle_merge_gap2instr() -> None:
    """Test `merge.gap2instr` function."""
    instr = merge.gap2instr(ARR_GAP)
    assert np.array_equal(instr, ARR_INSTR)


def test_subtitle_merge_merge_tl() -> None:
    """Test `merge.merge_tl` function."""
    tl = merge.merge_tl(ARR_TL, ARR_INSTR)
    assert np.array_equal(tl, ARR_TL_MERGED)


def test_subtitle_merge_merge_txt() -> None:
    """Test `merge.merge_txt` function."""
    txt = merge.merge_txt(ARR_TXT, ARR_INSTR)
    assert np.array_equal(txt, ARR_TXT_MERGED)
