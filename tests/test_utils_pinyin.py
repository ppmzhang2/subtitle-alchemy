"""Test pinyin similarity in utils."""

import numpy as np
from subalch.utils import pinyin

v1 = np.array(
    ["", "wth", "马", "关", "看", "发", "飞", "煤", "买", "大", "市", "饭"],
    dtype="U",
)

v2 = np.array(
    ["😯", "я", "吗", "光", "干", "飞", "高", "么", "外", "踏", "事", "方"],
    dtype="U",
)


def test_pinyin_get_ind_vec() -> None:
    """Test `pinyin.get_ind_vec` function."""
    ar_ind = pinyin.get_ind_vec(v1)
    ind_exp = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [3, 4, 3],
            [9, 22, 1],
            [10, 20, 4],
            [4, 4, 1],
            [4, 14, 1],
            [3, 14, 2],
            [3, 12, 3],
            [5, 4, 4],
            [17, 1, 4],
            [4, 20, 4],
        ],
        dtype=np.uint8,
    )
    assert np.all(ar_ind == ind_exp)


def test_pinyin_sim_vec() -> None:
    """Test `pinyin.sim_vec` function."""
    ar_sim = pinyin.sim_vec(v1, v2)
    sim_lower = np.array(
        [0, 0, 0.7, 0.7, 0.7, 0.3, 0.1, 0.6, 0.1, 0.7, 1, 0.5],
        dtype=np.float32,
    )
    sim_upper = np.array(
        [0, 0, 0.9, 0.9, 0.9, 0.7, 0.7, 0.9, 0.5, 0.9, 1, 0.9],
        dtype=np.float32,
    )
    assert np.all(ar_sim >= sim_lower)
    assert np.all(ar_sim <= sim_upper)
