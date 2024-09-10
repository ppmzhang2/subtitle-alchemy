"""Match the ground truth text with the predicted subtitles."""

import numpy as np

from subalch.utils import pinyin

CHR_DEFAULT = "✅"  # Placeholder for missing ground truth or prediction
IDX_DEFAULT = -1  # Placeholder index


def get_aligned_ind_raw(lab: np.ndarray, prd: np.ndarray) -> np.ndarray:
    """Get the raw aligned indices of predictions.

    Args:
        lab (list[str]): The 1D ground truth array of characters.
        prd (list[str]): The 1D predicted array of characters.

    Returns:
        np.ndarray: A list of indices representing the alignment between
          `lab` and `prd`. It has the same length as `lab`, where each entry
          corresponds to either an index in the predicted sequence `prd` or
          `IDX_DEFAULT` if the prediction is missing or incorrect.
          Note that falsely predicted characters are marked with `IDX_DEFAULT`,
          which is not the case for final output.
    """
    n_lab = len(lab)
    n_prd = len(prd)
    seq_prd_idx = list(range(n_prd))

    # DP table to store maximum matches
    dp = [[0] * (n_prd + 1) for _ in range(n_lab + 1)]

    # Fill the DP table
    for i in range(1, n_lab + 1):
        for j in range(1, n_prd + 1):
            if lab[i - 1] == prd[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtracking to determine the aligned prediction and indices
    # seq_chr_aligned = []
    seq_idx_aligned = []
    i_lab, i_prd = n_lab, n_prd
    while i_lab > 0 and i_prd > 0:
        if lab[i_lab - 1] == prd[i_prd - 1]:
            # Case: Match
            # seq_chr_aligned.append(seq_prd[i_prd - 1])
            seq_idx_aligned.append(seq_prd_idx[i_prd - 1])
            i_lab -= 1
            i_prd -= 1
        elif dp[i_lab - 1][i_prd] >= dp[i_lab][i_prd - 1]:
            # Case: missing prediction
            # seq_chr_aligned.append(CHR_DEFAULT)
            seq_idx_aligned.append(IDX_DEFAULT)
            i_lab -= 1
        else:
            # Skip this character in the prediction
            i_prd -= 1

    while i_lab > 0:
        # missing GT
        # seq_chr_aligned.append(CHR_DEFAULT)
        seq_idx_aligned.append(IDX_DEFAULT)
        i_lab -= 1

    # Reverse to get the correct order
    # seq_chr_aligned.reverse()
    seq_idx_aligned.reverse()

    return np.array(seq_idx_aligned, dtype=np.int32)


def locate_mismatch(
    idx: np.ndarray,
    n_prd: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Find indices of the mis-matched characters in GT and predicted text.

    It depends on the output of the `get_match_idx` function to identify the
    mis-matched characters indices in the ground truth and predicted sequences.

    The function returns two lists:
    - A list of lists, where each sublist contains the indices in the ground
      truth sequence that correspond to missing or incorrectly predicted
      characters.
    - A list of lists, where each sublist contains the indices in the predicted
      sequence that are potential candidates for the missing ground truth
      characters.

    Args:
        idx (np.ndarray): A list of indices representing the alignment between
          `seq_lab` and `seq_prd`. It has the same length as `seq_lab`, where
          each entry corresponds to either an index in the predicted sequence
          `seq_prd` or `IDX_DEFAULT` if the prediction is missing or incorrect.
        n_prd (int): The length of the predicted sequence.

    Returns:
        tuple[list[list[int]], list[list[int]]]: A tuple containing two lists:
        - `sseq_idx_lab_incorrect`: A list where each sublist contains the
          indices in `seq_lab` corresponding to ground truth characters that
          are missing or incorrectly predicted.
        - `sseq_idx_prd_incorrect`: A list where each sublist is a sequence of
          consecutive indices in `seq_prd`, corresponding to one or more ground
          truth characters that are missing or incorrectly predicted.

    Example:
        seq_lab = [..., "轻", "松", "地", "阅", "读", ...]
        seq_prd = [..., "轻", "速", "阿", "的", "阅", "读", ...]
        idx_aligned = [..., 79, -1, -1, 83, 84, ...]

        sseq_idx_lab_incorrect, sseq_idx_prd_incorrect = match_wrong_chrs(
                idx_aligned)

        # Output: [..., [77, 78], ...]; ["松", "地"]
        print(sseq_idx_lab_incorrect)
        # Output: [..., [80, 81, 82], ...]; ["速", "阿", "的"]
        print(sseq_idx_prd_incorrect)
    """
    sseq_idx_lab_incorrect = []
    sseq_idx_prd_incorrect = []
    _idx_lab_last_correct, _idx_prd_last_correct = -1, -1
    _flag_correct = True
    _seq_idx_lab_incorrect = []
    for idx_lab, idx_prd in enumerate(idx):
        if idx_prd == IDX_DEFAULT:
            _seq_idx_lab_incorrect.append(idx_lab)
            _flag_correct = False
        elif _flag_correct is False:
            sseq_idx_prd_incorrect.append(
                list(range(_idx_prd_last_correct + 1, idx_prd))
            )
            sseq_idx_lab_incorrect.append(_seq_idx_lab_incorrect)
            _seq_idx_lab_incorrect = []
            _idx_lab_last_correct, _idx_prd_last_correct = idx_lab, idx_prd
            _flag_correct = True
        else:
            _idx_lab_last_correct, _idx_prd_last_correct = idx_lab, idx_prd
            _flag_correct = True

    # If the last character is incorrect
    if _flag_correct is False:
        sseq_idx_lab_incorrect.append(_seq_idx_lab_incorrect)
        sseq_idx_prd_incorrect.append(
            list(range(_idx_prd_last_correct + 1, n_prd))
        )

    return sseq_idx_lab_incorrect, sseq_idx_prd_incorrect


def _map_lab2prd_miss(
    grp_lab: list[int],
    grp_prd: list[int],
    seq_lab: list[str],
    seq_prd: list[str],
) -> dict[int, int]:
    """Prediction missing characters, helper for `get_uniq_miss_map`."""
    _map = {}
    n_lab = len(grp_lab)
    n_prd = len(grp_prd)

    for i, idx_prd in enumerate(grp_prd):
        idx_lab_maxsim = None  # GT index with max similarity
        sim_max = 0.0
        prd_char = seq_prd[idx_prd]

        # do NOT calculate if out of order
        for j in range(i, i + n_lab - n_prd + 1):
            idx_lab = grp_lab[j]
            lab_char = seq_lab[idx_lab]
            sim_pinyin = pinyin.sim(lab_char, prd_char)

            if sim_pinyin > sim_max:
                sim_max = sim_pinyin
                idx_lab_maxsim = idx_lab

        if idx_lab_maxsim is not None:
            _map[idx_lab_maxsim] = idx_prd

    # loop through the GT
    for idx_lab in grp_lab:
        if idx_lab not in _map:
            _map[idx_lab] = -1

    return _map


def _map_lab2prd_extra(
    grp_lab: list[int],
    grp_prd: list[int],
    seq_lab: list[str],
    seq_prd: list[str],
) -> dict[int, int]:
    """Prediction extra characters, helper for `get_uniq_miss_map`."""
    _map = {}
    n_lab = len(grp_lab)
    n_prd = len(grp_prd)

    for i, idx_lab in enumerate(grp_lab):
        idx_prd_maxsim = None  # prediction index with max similarity
        sim_max = 0.0
        lab_char = seq_lab[idx_lab]

        # do NOT calculate if out of order
        for j in range(i, i + n_prd - n_lab + 1):
            idx_prd = grp_prd[j]
            prd_char = seq_prd[idx_prd]
            sim_pinyin = pinyin.sim(lab_char, prd_char)

            if sim_pinyin > sim_max:
                sim_max = sim_pinyin
                idx_prd_maxsim = idx_prd

        if idx_prd_maxsim is not None:
            _map[idx_lab] = idx_prd_maxsim

    return _map


def get_mismatch_map(
    seq_lab: list[str],
    seq_prd: list[str],
    sseq_idx_lab_incorrect: list[list[int]],
    sseq_idx_prd_incorrect: list[list[int]],
) -> dict[int, int]:
    """Get 1-to-1 mapping of incorrectly predicted GT index to prediction one.

    - The map is from the GT index to the prediction index.
    - There are three cases to consider:
      - Predicted wrong characters: in this case the 1-to-1 mapping is trivial.
      - Missing characters: the prediction is missing one or more characters.
        Note that in this case some GT indices will NOT have corresponding
        prediction indices.
      - Extra characters: the prediction has one or more extra characters.
    - Map incorrectly predicted GT Chinese characters to their closest
      predictions based on phonetic similarity, returning a dictionary from GT
      index to prediction index.

    Args:
        seq_lab (list[str]): The ground truth sequence of characters.
        seq_prd (list[str]): The predicted sequence of characters.
        sseq_idx_lab_incorrect (list[list[int]]): A list where each sublist
            contains the indices in `seq_lab` corresponding to ground truth
            characters that are missing or incorrectly predicted.
        sseq_idx_prd_incorrect (list[list[int]]): A list where each sublist
            contains the indices in `seq_prd` corresponding to one or more
            ground truth characters that are missing or incorrectly predicted.

    Returns:
        dict[int, int]: A dictionary mapping each incorrectly predicted GT
            index to its closest prediction index.
    """
    mapping = {}

    for lab_grp, prd_grp in zip(
        sseq_idx_lab_incorrect, sseq_idx_prd_incorrect, strict=True
    ):
        n_lab = len(lab_grp)
        n_prd = len(prd_grp)
        # 1. Predicted wrong characters
        # This is the simplest case where the 1-to-1 mapping is determined
        if n_prd == n_lab:
            for idx_lab, idx_prd in zip(lab_grp, prd_grp, strict=True):
                mapping[idx_lab] = idx_prd
        # 2. Missing characters
        elif n_prd < n_lab:
            _map_miss = _map_lab2prd_miss(lab_grp, prd_grp, seq_lab, seq_prd)
            mapping.update(_map_miss)

        # 3. Extra characters
        else:
            _map_extra = _map_lab2prd_extra(lab_grp, prd_grp, seq_lab, seq_prd)
            mapping.update(_map_extra)

    return mapping


def get_aligned_index(ar_lab: np.ndarray, ar_prd: np.ndarray) -> np.ndarray:
    """Get aligned indices of predictions.

    Args:
        ar_lab (np.ndarray): The ground truth sequence of characters.
        ar_prd (np.ndarray): The predicted sequence of characters.

    Returns:
        np.ndarray: A list of indices representing the alignment between
          `ar_lab` and `ar_prd`. It has the same length as `ar_lab`, where
          each entry corresponds to either an index in the predicted sequence
          `ar_prd` or `IDX_DEFAULT` if the prediction is missing. Falsely
          predicted character indices are also included.
    """
    _mat_match = {}
    idx_aligned_raw = get_aligned_ind_raw(ar_lab, ar_prd)
    for idx_lab, idx_prd in enumerate(idx_aligned_raw):
        if idx_prd != IDX_DEFAULT:
            _mat_match[idx_lab] = idx_prd
    sseq_idx_lab_mis, sseq_idx_prd_mis = locate_mismatch(
        idx_aligned_raw, len(ar_prd)
    )
    _map_mis = get_mismatch_map(
        ar_lab, ar_prd, sseq_idx_lab_mis, sseq_idx_prd_mis
    )
    _mat_match.update(_map_mis)
    return np.array([v for _, v in sorted(_mat_match.items())], dtype=np.int32)
