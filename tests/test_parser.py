"""Test subtitle_alchemy.parser module."""

import numpy as np
from subtitle_alchemy.parser import srt


def test_parse_srt() -> None:
    """Test parsing SRT string into arrays."""
    srt_string = (
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "First line\n"
        "\n"
        "2\n"
        "00:00:02,500 --> 00:00:03,500\n"
        "Second line\n"
        "continues here\n"
        "\n"
        "3\n"
        "00:01:00,000 --> 00:01:30,000\n"
        "Third line\n"
    )

    ar_tx, ar_ts = srt.generate(srt_string)

    # Check lines array
    ar_tx_exp = np.array(
        ["First line", "Second line continues here", "Third line"]
    )

    # Check timestamps array
    ar_ts_exp = np.array([[1000, 2000], [2500, 3500], [60000, 90000]])

    assert np.array_equal(ar_ts, ar_ts_exp)
    assert np.array_equal(ar_tx, ar_tx_exp)
