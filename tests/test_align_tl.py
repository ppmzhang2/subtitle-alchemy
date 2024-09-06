"""Test script alignment module."""

import numpy as np
from subalch.align._tl import acc_default
from subalch.align._tl import cum_default
from subalch.align._tl import get_aligned_tl
from subalch.align._tl import get_match_tl
from subalch.align._tl import get_miss_tl
from subalch.align._tl import impute_pre
from subalch.align._tl import impute_suc

# Example usage:
INDICES_ALIGNED = np.array(
    [0, -1, 1, 2, 3, 4, 5, -1, -1, -1, 7, 8, 9], dtype=np.int32
)
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
    ],
    dtype=np.uint32,
)


def test_align_acc_default() -> None:
    """Test `acc_default` function."""
    acc = acc_default(INDICES_ALIGNED)
    acc_exp = np.array([0, 1, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0], dtype=np.int32)
    assert np.array_equal(acc, acc_exp)


def test_align_cum_default() -> None:
    """Test `cum_default` function."""
    cum = cum_default(acc_default(INDICES_ALIGNED))
    cum_exp = np.array([0, 1, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0], dtype=np.int32)
    assert np.array_equal(cum, cum_exp)


def test_align_impute_pre() -> None:
    """Test `impute_pre` function."""
    idx_preimputed = impute_pre(INDICES_ALIGNED)
    idx_preimputed_exp = np.array(
        [0, 0, 1, 2, 3, 4, 5, 5, 5, 5, 7, 8, 9], dtype=np.int32
    )
    assert np.array_equal(idx_preimputed, idx_preimputed_exp)


def test_align_impute_suc() -> None:
    """Test `impute_suc` function."""
    idx_sucimputed = impute_suc(INDICES_ALIGNED)
    idx_sucimputed_exp = np.array(
        [0, 1, 1, 2, 3, 4, 5, 7, 7, 7, 7, 8, 9], dtype=np.int32
    )
    assert np.array_equal(idx_sucimputed, idx_sucimputed_exp)


def test_align_get_miss_tl() -> None:
    """Test `get_miss_tl` function."""
    tl_miss = get_miss_tl(INDICES_ALIGNED, ARR_TL)
    tl_miss_exp = np.array(
        [[1, 100, 300], [7, 1735, 1756], [8, 1756, 1777], [9, 1777, 1800]],
        dtype=np.uint32,
    )
    assert np.array_equal(tl_miss, tl_miss_exp)


def test_align_get_match_tl() -> None:
    """Test `get_match_tl` function."""
    tl_match = get_match_tl(INDICES_ALIGNED, ARR_TL)
    tl_match_exp = np.array(
        [
            [0, 0, 100],
            [2, 300, 800],
            [3, 1100, 1190],
            [4, 1200, 1230],
            [5, 1250, 1300],
            [6, 1700, 1735],
            [10, 1800, 1870],
            [11, 1900, 1910],
            [12, 2410, 2510],
        ],
        dtype=np.uint32,
    )
    assert np.array_equal(tl_match, tl_match_exp)


def test_align_get_aligned_tl() -> None:
    """Test `get_aligned_tl` function."""
    tl_aligned = get_aligned_tl(INDICES_ALIGNED, ARR_TL)
    tl_aligned_exp = np.array(
        [
            [0, 100],
            [100, 300],
            [300, 800],
            [1100, 1190],
            [1200, 1230],
            [1250, 1300],
            [1700, 1735],
            [1735, 1756],
            [1756, 1777],
            [1777, 1800],
            [1800, 1870],
            [1900, 1910],
            [2410, 2510],
        ],
        dtype=np.uint32,
    )
    assert np.array_equal(tl_aligned, tl_aligned_exp)
