"""Align timeline with alignment index."""

import numpy as np


def acc_default(idx: np.ndarray) -> np.ndarray:
    """Get an array with the count of accumulated default values.

    Example:
        input:  [0, -1, 1, 2, 3, 4, 5, -1, -1, -1, 7, 8, 9]
        output: [0,  1, 0, 0, 0, 0, 0,  1,  2,  3, 0, 0, 0]

    Args:
        idx (np.ndarray): the alignment index with sparse -1 values.

    Returns:
        np.ndarray: the alignment index with -1 values imputed with its
            accumulated count.
    """
    # Initialize the output array with zeros
    ret = np.zeros_like(idx, dtype=np.int32)

    # Variable to track the count of consecutive `-1`s
    count = 0

    # Iterate over the array
    for i in range(idx.size):
        if idx[i] == -1:
            count += 1
            ret[i] = count  # Set the output to the count
        else:
            count = 0  # Reset count when it's not a `-1`

    return ret


def cum_default(acc: np.ndarray) -> np.ndarray:
    """Get an array with the total count of consecutive default values.

    Note that the input should be the output of the `acc_default` function.

    """
    ret = np.zeros_like(acc)

    # Track the current maximum in the increasing series
    max_cur = 0

    # Iterate through the array backwards
    for i in range(acc.size - 1, -1, -1):
        if acc[i] == 0:
            # Reset the current max when we encounter a 0
            max_cur = 0
        else:
            # Update the current max (it keeps the largest value in the series)
            max_cur = max(max_cur, acc[i])
            # Store the current max in the result array
            ret[i] = max_cur

    return ret


def impute_pre(idx: np.ndarray) -> np.ndarray:
    """Impute negative values with the its predecessor.

    Args:
        idx (np.ndarray): the alignment index with sparse -1 values.

    Returns:
        np.ndarray: the alignment index with -1 values imputed with its
            predecessor.
    """
    # copy the input array
    idx = idx.copy()

    # Get the indices where the elements are default values (-1)
    indices_default = np.where(idx == -1)[0]

    # Iterate over the indices and replace -1 with the previous element
    for i in indices_default:
        if i > 0:  # Ensure there is a predecessor
            idx[i] = idx[i - 1]

    return idx


def impute_suc(idx: np.ndarray) -> np.ndarray:
    """Impute negative values with the its successors.

    Args:
        idx (np.ndarray): the alignment index with sparse -1 values.

    Returns:
        np.ndarray: the alignment index with -1 values imputed with its
            successors.
    """
    # copy the input array
    arr = idx.copy()

    # Get the indices where the elements are -1
    indices_default = np.where(arr == -1)[0]

    # Iterate backwards over the indices and replace -1 with the next element
    for i in reversed(indices_default):
        if i < len(arr) - 1:  # Ensure there is a successor
            arr[i] = arr[i + 1]

    return arr


def get_miss_interval(
    acc: np.ndarray,
    cum: np.ndarray,
    pre: np.ndarray,
    suc: np.ndarray,
    tl: np.ndarray,
) -> np.ndarray:
    """Get the missing timeline intervals, and other necessary information."""
    # Get the indices of default values
    indices_diff = np.where(cum >= 1)[0]
    n_diff = indices_diff.size
    interval_diff = np.zeros((n_diff, 5), dtype=np.int32)
    interval_diff[:, 0] = acc[indices_diff]
    interval_diff[:, 1] = cum[indices_diff]
    interval_diff[:, 4] = indices_diff
    for k, v in enumerate(indices_diff):
        interval_diff[k, 2] = tl[pre[v], 1]
        interval_diff[k, 3] = tl[suc[v], 0]
    return interval_diff


def _get_miss_tl_start(idx: int, cnt: int, start: int, end: int) -> int:
    """Get the start time of the missing timeline.

    Note: need to be vectorized.
    """
    # only one missing, or the first missing
    if cnt == 1 or idx == 1:
        return start
    intvl = (end - start) // cnt
    return start + (idx - 1) * intvl


def _get_miss_tl_end(idx: int, cnt: int, start: int, end: int) -> int:
    """Get the end time of the missing timeline.

    Note: need to be vectorized.
    """
    # only one missing, or the last missing
    if cnt in (1, idx):
        return end
    intvl = (end - start) // cnt
    return start + idx * intvl


def get_miss_tl(idx_aligned: np.ndarray, tl: np.ndarray) -> np.ndarray:
    """Interpolate the missing timeline.

    To interpolate the missing timeline, we need to know:
    - the earliest possible start time (end of the previous matched timeline)
    - the latest possible end time (start of the next matched timeline)
    - if it belongs to a consecutive missing timeline, we also need to know:
      - total number of missing timelines
      - index of this missing item in the consecutive missing series
    We approach this using the most naive way: separate the longest possible
    interval evenly before assignment.

    Args:
        idx_aligned (np.ndarray): the 1D alignment index array (N, ) with
            sparse default values.
        tl (np.ndarray): the 2D timeline array (N, 2) with start and end time
            pairs.

    Returns:
        np.ndarray: 2D array (N, 3) with the missing timelines:
        - (N, 0): the index of `N`, indicating the missing location.
        - (N, 1): the start time of the missing timeline.
        - (N, 2): the end time of the missing timeline.
    """
    # TODO: move acc_default and cum_default to `_idx.py`
    acc = acc_default(idx_aligned)
    cum = cum_default(acc)

    idx_pre = impute_pre(idx_aligned)
    idx_suc = impute_suc(idx_aligned)
    # TODO: if the missing interval is too long, it is possible that the
    #   the content was discarded by the speaker. Consider to add a threshold,
    #   and and a function to align the text.
    intvl = get_miss_interval(acc, cum, idx_pre, idx_suc, tl)
    tl_start = np.vectorize(_get_miss_tl_start, otypes=[np.int32])(
        intvl[:, 0], intvl[:, 1], intvl[:, 2], intvl[:, 3]
    )
    tl_end = np.vectorize(_get_miss_tl_end, otypes=[np.int32])(
        intvl[:, 0], intvl[:, 1], intvl[:, 2], intvl[:, 3]
    )
    return np.column_stack((intvl[:, 4], tl_start, tl_end))


def get_match_tl(idx_aligned: np.ndarray, tl: np.ndarray) -> np.ndarray:
    """Get the matched timeline.

    Args:
        idx_aligned (np.ndarray): the 1D alignment index array (N, ) with
            sparse default values.
        tl (np.ndarray): the 2D timeline array (N, 2) with start and end time
            pairs.

    Returns:
        np.ndarray: 2D array (N, 3) with the matched timelines:
        - (N, 0): the index of `N`, indicating the matched location.
        - (N, 1): the start time of the matched timeline.
        - (N, 2): the end time of the matched timeline.
    """
    idx = np.where(idx_aligned != -1)[0]
    tl_ = tl[idx_aligned[idx]]
    return np.column_stack((idx, tl_))


def get_aligned_tl(idx_aligned: np.ndarray, tl: np.ndarray) -> np.ndarray:
    """Get the aligned timeline.

    Args:
        idx_aligned (np.ndarray): the 1D alignment index array (N, ) with
            sparse default values.
        tl (np.ndarray): the 2D timeline array (N, 2) with start and end time
            pairs.

    Returns:
        np.ndarray: 2D array (N, 3) with the aligned timelines:
        - (N, 0): the index of `N`, indicating the aligned location.
        - (N, 1): the start time of the aligned timeline.
        - (N, 2): the end time of the aligned timeline.
    """
    tl_miss = get_miss_tl(idx_aligned, tl)
    tl_match = get_match_tl(idx_aligned, tl)
    tl = np.concatenate((tl_miss, tl_match), axis=0)
    return tl[tl[:, 0].argsort()][:, 1:].astype(np.int32)
