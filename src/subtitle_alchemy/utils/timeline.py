"""Timeline utilities."""

import numpy as np


def ms2hmss(tl: np.ndarray) -> np.ndarray:
    """Convert the milliseconds timeline array to HH:MM:SS.mmm array.

    Args:
        tl (np.ndarray): 2D array (N, 2) of milliseconds

    Returns:
        np.ndarray: 2D array (N, 8) timeline i.e. hours (start), hours (end),
        minutes (start), minutes (end), seconds (start), seconds (end),
        milliseconds (start), milliseconds (end)
    """
    return np.column_stack(
        (
            tl // 3600000,  # hours
            tl // 60000 % 60,  # minutes
            tl // 1000 % 60,  # seconds
            tl % 1000,  # milliseconds
        )
    )


def hmss2ms(hr: int, mn: int, ss: int, ms: int) -> int:
    """Convert the HH:MM:SS.mmm to milliseconds.

    Args:
        hr (int): hours
        mn (int): minutes
        ss (int): seconds
        ms (int): milliseconds

    Returns:
        int: milliseconds
    """
    return hr * 3600000 + mn * 60000 + ss * 1000 + ms
