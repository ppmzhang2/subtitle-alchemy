"""Separate text (Chinese characters and ASCII words) with punctuation."""
# ruff: noqa: RUF001

import re

import numpy as np

# Regular expression to match Chinese characters
PATTERN_ZH = r"[\u4e00-\u9fff]"

# Regular expression to match ASCII words (sequences of A-Z, a-z, 0-9)
PATTERN_ASCII = r"[a-zA-Z0-9]+"

# Combined word patterns to match Chinese characters or ASCII words
PATTERN_WORD = f"({PATTERN_ZH})|({PATTERN_ASCII})"

_K_NONE = 0  # No punctuation
# Dictionary to map punctuation marks to integers
MAP_PUNC2INT = {
    "，": 1,  # Chinese comma
    "。": 2,  # Chinese period
    "！": 3,  # Chinese exclamation mark
    "？": 4,  # Chinese question mark
    "、": 5,  # Chinese enumeration mark
    "；": 6,  # Chinese semicolon
    "：": 7,  # Chinese colon
    "“": 8,  # Chinese left double quotation mark
    "”": 9,  # Chinese right double quotation mark
    "‘": 10,  # Chinese left single quotation mark
    "’": 11,  # Chinese right single quotation mark
    "「": 12,  # Chinese left corner bracket
    "」": 13,  # Chinese right corner bracket
    "『": 14,  # Chinese left white corner bracket
    "』": 15,  # Chinese right white corner bracket
    "《": 16,  # Chinese left double angle bracket
    "》": 17,  # Chinese right double angle bracket
    "（": 18,  # Chinese left parenthesis
    "）": 19,  # Chinese right parenthesis
    "【": 20,  # Chinese left square bracket
    "】": 21,  # Chinese right square bracket
    "〈": 22,  # Chinese left angle bracket
    "〉": 23,  # Chinese right angle bracket
    ",": 24,  # English comma
    ".": 25,  # English period
    "!": 26,  # English exclamation mark
    "?": 27,  # English question mark
}
MAP_INT2PUNC = {v: k for k, v in MAP_PUNC2INT.items()}


def separate(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Separate text (Chinese characters and ASCII words) with punctuation.

    TODO | NOTE: If there are multiple punctuation marks consecutively
    (e.g., !?!), the script only identifies the first one.

    Args:
        text (str): The input string containing Chinese characters, ASCII words
            and punctuation marks.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
        - A UTF-8 encoded 1D array of strings where each element is a Chinese
          character or ASCII word.
        - A 1D uint16 array of integers where each element:
          - 0 if there's no punctuation after the corresponding element
          - a positive integer if there is punctuation (1: comma, 2: period,
            etc.).
    """
    len_text = len(text)
    # Find all matches in the text (including the index positions)
    words = re.finditer(PATTERN_WORD, text)

    # Lists to store results and punctuation info
    seq_txt = []
    seq_mark = []

    # Iterate over all matches to extract the words/characters and check for
    # punctuation
    for word in words:
        # Append the matched text (either a Chinese character or ASCII word)
        seq_txt.append(word.group())

        # Check the character right after the current match
        match_end = word.end()
        if match_end < len_text and text[match_end] in MAP_PUNC2INT:
            # If the next character is punctuation, append its mapped value
            seq_mark.append(MAP_PUNC2INT[text[match_end]])
        else:
            # Otherwise no punctuation
            seq_mark.append(_K_NONE)

    return np.array(seq_txt, dtype="U"), np.array(seq_mark, dtype=np.uint16)


def restore(text: np.ndarray, punc: np.ndarray) -> np.ndarray:
    """Restore punctuation marks to the text.

    Args:
        text (np.ndarray): A 1D UTF-8 encoded array (N, ) of strings where each
            element is a Chinese character or an ASCII word.
        punc (np.ndarray): A 1D uint16 array of integers where each element:
            - 0 if there's no punctuation after the corresponding element
            - a positive integer if there is punctuation (1: comma, 2: period,
              etc.).

    Returns:
        np.ndarray: A 1D UTF-8 encoded array (N, ) of strings where each
        element is a Chinese character or an ASCII word, which may be followed
        by a punctuation mark.
    """
    # Initialize an empty list to store the restored text
    restored = []

    # Iterate over the text and punctuation arrays
    for _, (word, mark) in enumerate(zip(text, punc, strict=True)):
        if mark == _K_NONE:
            # If there's no punctuation, append the word/character
            restored.append(word)
        else:
            # Otherwise append punctuation mark to the word and then append
            # to the list
            restored.append(word + MAP_INT2PUNC[mark])

    return np.array(restored, dtype="U")
