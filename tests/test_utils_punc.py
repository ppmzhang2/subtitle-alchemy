"""Test pinyin similarity in utils."""
# ruff: noqa: RUF001

import numpy as np
from subtitle_alchemy.utils import punc

TEXT = """
你好，World！这是一、二个测试。Hello, 世界! Testing123.
"""
ARR_TXT_EXP = np.array(
    [
        "你",
        "好",
        "World",
        "这",
        "是",
        "一",
        "二",
        "个",
        "测",
        "试",
        "Hello",
        "世",
        "界",
        "Testing123",
    ],
    dtype="U",
)
ARR_PUNC_EXP = np.array(
    [0, 1, 3, 0, 0, 5, 0, 0, 0, 2, 24, 0, 26, 25], dtype=np.uint16
)
ARR_TXTPUNC_EXP = np.array(
    [
        "你",
        "好，",
        "World！",
        "这",
        "是",
        "一、",
        "二",
        "个",
        "测",
        "试。",
        "Hello,",
        "世",
        "界!",
        "Testing123.",
    ],
    dtype="U",
)


def test_utils_punc_separate() -> None:
    """Test `punc.separate`."""
    arr_text, arr_punc = punc.separate(TEXT.strip())
    assert np.array_equal(arr_text, ARR_TXT_EXP)
    assert np.array_equal(arr_punc, ARR_PUNC_EXP)


def test_utils_punc_restore() -> None:
    """Test `punc.restore`."""
    arr_txtpunc = punc.restore(ARR_TXT_EXP, ARR_PUNC_EXP)
    assert np.array_equal(arr_txtpunc, ARR_TXTPUNC_EXP)
