"""Match the ground truth text with the predicted subtitles."""

CHR_DEFAULT = "✅"  # Placeholder for missing ground truth or prediction
IDX_DEFAULT = -1  # Placeholder index


def get_match_idx(seq_lab: list[str], seq_prd: list[str]) -> list[int]:
    """Get the aligned indices between the GT and predicted scripts text.

    Args:
        seq_lab (list[str]): The ground truth sequence of characters.
        seq_prd (list[str]): The predicted sequence of characters.

    Returns:
        list[int]: A list of indices representing the alignment between
          `seq_lab` and `seq_prd`. It has the same length as `seq_lab`, where
          each entry corresponds to either an index in the predicted sequence
          `seq_prd` or `IDX_DEFAULT` if the prediction is missing or incorrect.
    """
    n_lab = len(seq_lab)
    n_prd = len(seq_prd)
    seq_prd_idx = list(range(n_prd))

    # DP table to store maximum matches
    dp = [[0] * (n_prd + 1) for _ in range(n_lab + 1)]

    # Fill the DP table
    for i in range(1, n_lab + 1):
        for j in range(1, n_prd + 1):
            if seq_lab[i - 1] == seq_prd[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtracking to determine the aligned prediction and indices
    # seq_chr_aligned = []
    seq_idx_aligned = []
    i_lab, i_prd = n_lab, n_prd
    while i_lab > 0 and i_prd > 0:
        if seq_lab[i_lab - 1] == seq_prd[i_prd - 1]:
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

    return seq_idx_aligned


def get_miss_idx(idx: list[int]) -> tuple[list[list[int]], list[list[int]]]:
    """Get indices of incorrectly prediction in the prediction and GT.

    It depends on the output of the `align_stt_prediction` function to
    identify the mis-matched characters indices in the ground truth and
    predicted sequences.

    The function returns two lists:
    - A list of lists, where each sublist contains the indices in the ground
      truth sequence that correspond to missing or incorrectly predicted
      characters.
    - A list of lists, where each sublist contains the indices in the predicted
      sequence that are potential candidates for the missing ground truth
      characters.

    Args:
        idx (list[int]): A list of indices representing the alignment between
          `seq_lab` and `seq_prd`. It has the same length as `seq_lab`, where
          each entry corresponds to either an index in the predicted sequence
          `seq_prd` or `IDX_DEFAULT` if the prediction is missing or incorrect.

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
                list(range(_idx_prd_last_correct + 1, idx_prd)))
            sseq_idx_lab_incorrect.append(_seq_idx_lab_incorrect)
            _seq_idx_lab_incorrect = []
            _idx_lab_last_correct, _idx_prd_last_correct = idx_lab, idx_prd
            _flag_correct = True
        else:
            _idx_lab_last_correct, _idx_prd_last_correct = idx_lab, idx_prd
            _flag_correct = True

    return sseq_idx_lab_incorrect, sseq_idx_prd_incorrect
