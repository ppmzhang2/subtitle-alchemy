"""Merge Subtitles.

There are two important types of arrays for merging control:

- Gap array: 1D array with only 0s and 1s of shape (N, ) which indicates
  either to merge (0) or not (1) with the next item.
  - The data type is uint8.
  - The last item is always 1.
  - It can be obtained easily with for example the timeline array or the
    punctuation array, but not straight-forward enough to do the merging, so we
    need to convert it to the instruction array.
- Instruction array: 2D array of shape (M, 2) which contains the start and end
  indices to merge. It can be obtained by the gap array.
"""

import numpy as np

_K_GAP_V_M = 0  # Gap value for merging
_K_GAP_V_S = 1  # Gap value for splitting

__all__ = ["tl2gap", "gap2instr", "merge_tl", "merge_txt"]


def tl2gap(tl: np.ndarray, th: int = 500) -> np.ndarray:
    """Compute the gap between each timestamp pair.

    Args:
        tl (np.ndarray): 2D array (N, 2) of start and end timelines
        th (int): The threshold in milliseconds to determine the gap, default
            is 500ms

    Returns:
        np.ndarray: 1D array (N, ) gap array for computing merge instructions.
            One `1` must be appended to the end of the array.
    """
    gap_ms = tl[1:, 0] - tl[:-1, 1]
    gap = np.where(gap_ms <= th, _K_GAP_V_M, _K_GAP_V_S)
    return np.append(gap, _K_GAP_V_S).astype(np.uint8)


def gap2instr(gap: np.ndarray) -> np.ndarray:
    """Get merge instructions from a (N, ) gap array.

    Args:
        gap (np.ndarray): 1D array (N, ) of 0s and 1s

    Returns:
        np.ndarray: (M, 2) merge instruction array which contains the start and
            end indices to merge
    """
    # Initialize list to store result pairs
    result = []
    lo = 0

    # Iterate through the array
    for idx, val in enumerate(gap):
        # If the 1st value is 1, add a pair of 0s
        if val == _K_GAP_V_S and idx == 0:
            result.append([0, 0])
            lo = idx + 1
        # If the current value and the previous value are 1, add a pair
        elif val == _K_GAP_V_S and idx > 0 and gap[idx - 1] == _K_GAP_V_S:
            result.append([idx, idx])
            lo = idx + 1
        # Consecutive 0s ended
        elif val == _K_GAP_V_S and idx > 0 and gap[idx - 1] == 0:
            result.append([lo, idx])
            lo = idx + 1
        # Ignore consecutive 0s
        else:
            continue

    return np.array(result, dtype=np.int32)


def merge_tl(tl: np.ndarray, instr: np.ndarray) -> np.ndarray:
    """Merge the timeline array based on a merge instruction array.

    Args:
        tl (np.ndarray): 2D array (N, 2) of start and end in milliseconds
        instr (np.ndarray): 2D array (M, 2) merge instruction array which
          contains the start and end indices to merge

    Returns:
        np.ndarray: 2D array (M, 2) of merged start and end timestamps
    """
    return np.column_stack((tl[instr[:, 0], 0], tl[instr[:, 1], 1]))


def merge_txt(txt: np.ndarray, instr: np.ndarray) -> np.ndarray:
    """Merge the text array based on a merge instruction array.

    Args:
        txt (np.ndarray): 1D array of UTF-8 encoded strings
        instr (np.ndarray): 2D array (M, 2) merge instruction array which
          contains the start and end indices to merge

    Returns:
        np.ndarray: 1D array (M, ) of merged strings
    """
    return np.array(["".join(txt[s : e + 1]) for s, e in instr])
