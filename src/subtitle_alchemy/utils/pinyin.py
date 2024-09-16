"""Compare the phonetic similarity between two Chinese characters."""

from __future__ import annotations

import numpy as np
from pypinyin import Style
from pypinyin import lazy_pinyin

_WEIGHT_INITIAL = 0.35
_WEIGHT_FINAL = 0.5
_WEIGHT_TONE = 0.15

_MAP_STR2INIT = {
    "b": 1,
    "p": 2,
    "m": 3,
    "f": 4,
    "d": 5,
    "t": 6,
    "n": 7,
    "l": 8,
    "g": 9,
    "k": 10,
    "h": 11,
    "j": 12,
    "q": 13,
    "x": 14,
    "zh": 15,
    "ch": 16,
    "sh": 17,
    "r": 18,
    "z": 19,
    "c": 20,
    "s": 21,
    "": 22,
}


_MAP_STR2FINAL = {
    "i": 1,
    "u": 2,
    "v": 3,
    "a": 4,
    "ia": 5,
    "ua": 6,
    "o": 7,
    "uo": 8,
    "e": 9,
    "ie": 10,
    "ve": 11,
    "ai": 12,
    "uai": 13,
    "ei": 14,
    "uei": 15,
    "ao": 16,
    "iao": 17,
    "ou": 18,
    "iou": 19,
    "an": 20,
    "ian": 21,
    "uan": 22,
    "van": 23,
    "en": 24,
    "in": 25,
    "uen": 26,
    "vn": 27,
    "ang": 28,
    "iang": 29,
    "uang": 30,
    "eng": 31,
    "ing": 32,
    "ueng": 33,
    "ong": 34,
    "iong": 35,
    "er": 36,
    "": 37,
    # TODO: "ê"
}

_MAP_STR2TONE = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
}


# Similarity matrix for Chinese Pinyin initials (consonants).
#
# Each row and column corresponds to an initial sound, and the value at the
# intersection of a row and column represents the phonetic similarity between
# the two initials.
# Values range from 0 (no similarity) to 1 (identical sounds).
# The matrix takes into account how closely related different sounds are based
# on their place of articulation and manner of articulation.
#
# - Higher values (e.g., 0.6-0.7) indicate stronger phonetic similarity between
#   sounds.
#   Labial sounds (b, p, m, f), retroflex sounds (zh, ch, sh, r),
#   dental/alveolar sounds (d, t, n, l, z, c, s), and velar sounds (g, k, h)
#   have higher internal similarities.
# - Moderate values (e.g., 0.4-0.5) represent similarities between sounds that
#   share some phonetic traits, like place of articulation, but differ in their
#   manner (e.g., `b` and `f`, or `g` and `h`).
# - Weak similarities (e.g., 0.1-0.3) represent sounds that are more distant,
#   but may still share some articulatory features.
# - `Initial.NONE` is given a low but non-zero similarity (0.15) with other
#   initials to account for cases where no initial sound is present in the
#   Pinyin syllable.
#
# This matrix helps in calculating phonetic similarity by incorporating partial
# matches between sounds, rather than treating them as entirely dissimilar.
SIM_MAT_INITIAL = np.array(
    [
        # DEFAULT
        [
            0,  # DEFAULT
            0,  # b
            0,  # p
            0,  # m
            0,  # f
            0,  # d
            0,  # t
            0,  # n
            0,  # l
            0,  # g
            0,  # k
            0,  # h
            0,  # j
            0,  # q
            0,  # x
            0,  # zh
            0,  # ch
            0,  # sh
            0,  # r
            0,  # z
            0,  # c
            0,  # s
            0,  # NONE
        ],
        # b
        [
            0.00,  # DEFAULT
            1.00,  # b
            0.70,  # p
            0.50,  # m
            0.40,  # f
            0.20,  # d
            0.20,  # t
            0.20,  # n
            0.20,  # l
            0.10,  # g
            0.10,  # k
            0.10,  # h
            0.20,  # j
            0.20,  # q
            0.10,  # x
            0.10,  # zh
            0.10,  # ch
            0.10,  # sh
            0.10,  # r
            0.20,  # z
            0.20,  # c
            0.10,  # s
            0.15,  # NONE
        ],
        # p
        [
            0.00,  # DEFAULT
            0.70,  # b
            1.00,  # p
            0.50,  # m
            0.40,  # f
            0.20,  # d
            0.20,  # t
            0.20,  # n
            0.20,  # l
            0.10,  # g
            0.10,  # k
            0.10,  # h
            0.20,  # j
            0.20,  # q
            0.10,  # x
            0.10,  # zh
            0.10,  # ch
            0.10,  # sh
            0.10,  # r
            0.20,  # z
            0.20,  # c
            0.10,  # s
            0.15,
        ],
        # m
        [
            0.00,  # DEFAULT
            0.50,  # b
            0.50,  # p
            1.00,  # m
            0.30,  # f
            0.10,  # d
            0.10,  # t
            0.20,  # n
            0.20,  # l
            0.10,  # g
            0.10,  # k
            0.10,  # h
            0.10,  # j
            0.10,  # q
            0.10,  # x
            0.10,  # zh
            0.10,  # ch
            0.10,  # sh
            0.10,  # r
            0.10,  # z
            0.10,  # c
            0.10,  # s
            0.15,
        ],
        # f
        [
            0.00,  # DEFAULT
            0.40,  # b
            0.40,  # p
            0.30,  # m
            1.00,  # f
            0.10,  # d
            0.10,  # t
            0.10,  # n
            0.10,  # l
            0.20,  # g
            0.20,  # k
            0.20,  # h
            0.10,  # j
            0.10,  # q
            0.10,  # x
            0.10,  # zh
            0.10,  # ch
            0.20,  # sh
            0.20,  # r
            0.20,  # z
            0.20,  # c
            0.20,  # s
            0.15,
        ],
        # d
        [
            0.00,  # DEFAULT
            0.20,  # b
            0.20,  # p
            0.10,  # m
            0.10,  # f
            1.00,  # d
            0.70,  # t
            0.50,  # n
            0.30,  # l
            0.20,  # g
            0.20,  # k
            0.20,  # h
            0.10,  # j
            0.10,  # q
            0.10,  # x
            0.10,  # zh
            0.10,  # ch
            0.10,  # sh
            0.10,  # r
            0.20,  # z
            0.20,  # c
            0.10,  # s
            0.15,
        ],
        # t
        [
            0.00,  # DEFAULT
            0.20,  # b
            0.20,  # p
            0.10,  # m
            0.10,  # f
            0.70,  # d
            1.00,  # t
            0.50,  # n
            0.30,  # l
            0.20,  # g
            0.20,  # k
            0.20,  # h
            0.10,  # j
            0.10,  # q
            0.10,  # x
            0.10,  # zh
            0.10,  # ch
            0.10,  # sh
            0.10,  # r
            0.20,  # z
            0.20,  # c
            0.10,  # s
            0.15,
        ],
        # n
        [
            0.00,  # DEFAULT
            0.20,  # b
            0.20,  # p
            0.20,  # m
            0.10,  # f
            0.50,  # d
            0.50,  # t
            1.00,  # n
            0.60,  # l
            0.20,  # g
            0.20,  # k
            0.20,  # h
            0.10,  # j
            0.10,  # q
            0.10,  # x
            0.10,  # zh
            0.10,  # ch
            0.10,  # sh
            0.20,  # r
            0.20,  # z
            0.20,  # c
            0.10,  # s
            0.15,
        ],
        # l
        [
            0.00,  # DEFAULT
            0.20,  # b
            0.20,  # p
            0.20,  # m
            0.10,  # f
            0.30,  # d
            0.30,  # t
            0.60,  # n
            1.00,  # l
            0.20,  # g
            0.20,  # k
            0.20,  # h
            0.10,  # j
            0.10,  # q
            0.10,  # x
            0.10,  # zh
            0.10,  # ch
            0.10,  # sh
            0.20,  # r
            0.20,  # z
            0.20,  # c
            0.20,  # s
            0.15,
        ],
        # g
        [
            0.00,  # DEFAULT
            0.10,  # b
            0.10,  # p
            0.10,  # m
            0.20,  # f
            0.20,  # d
            0.20,  # t
            0.20,  # n
            0.20,  # l
            1.00,  # g
            0.70,  # k
            0.50,  # h
            0.10,  # j
            0.10,  # q
            0.10,  # x
            0.10,  # zh
            0.10,  # ch
            0.10,  # sh
            0.10,  # r
            0.10,  # z
            0.10,  # c
            0.10,  # s
            0.15,
        ],
        # k
        [
            0.00,  # DEFAULT
            0.10,  # b
            0.10,  # p
            0.10,  # m
            0.20,  # f
            0.20,  # d
            0.20,  # t
            0.20,  # n
            0.20,  # l
            0.70,  # g
            1.00,  # k
            0.50,  # h
            0.10,  # j
            0.10,  # q
            0.10,  # x
            0.10,  # zh
            0.10,  # ch
            0.10,  # sh
            0.10,  # r
            0.10,  # z
            0.10,  # c
            0.10,  # s
            0.15,
        ],
        # h
        [
            0.00,  # DEFAULT
            0.10,  # b
            0.10,  # p
            0.10,  # m
            0.20,  # f
            0.20,  # d
            0.20,  # t
            0.20,  # n
            0.20,  # l
            0.50,  # g
            0.50,  # k
            1.00,  # h
            0.10,  # j
            0.10,  # q
            0.10,  # x
            0.10,  # zh
            0.10,  # ch
            0.10,  # sh
            0.10,  # r
            0.20,  # z
            0.20,  # c
            0.20,  # s
            0.15,
        ],
        # j
        [
            0.00,  # DEFAULT
            0.20,  # b
            0.20,  # p
            0.10,  # m
            0.10,  # f
            0.10,  # d
            0.10,  # t
            0.10,  # n
            0.10,  # l
            0.10,  # g
            0.10,  # k
            0.10,  # h
            1.00,  # j
            0.70,  # q
            0.60,  # x
            0.20,  # zh
            0.20,  # ch
            0.20,  # sh
            0.10,  # r
            0.20,  # z
            0.20,  # c
            0.10,  # s
            0.15,
        ],
        # q
        [
            0.00,  # DEFAULT
            0.20,  # b
            0.20,  # p
            0.10,  # m
            0.10,  # f
            0.10,  # d
            0.10,  # t
            0.10,  # n
            0.10,  # l
            0.10,  # g
            0.10,  # k
            0.10,  # h
            0.70,  # j
            1.00,  # q
            0.65,  # x
            0.20,  # zh
            0.20,  # ch
            0.20,  # sh
            0.10,  # r
            0.20,  # z
            0.20,  # c
            0.10,  # s
            0.15,
        ],
        # x
        [
            0.00,  # DEFAULT
            0.10,  # b
            0.10,  # p
            0.10,  # m
            0.10,  # f
            0.10,  # d
            0.10,  # t
            0.10,  # n
            0.10,  # l
            0.10,  # g
            0.10,  # k
            0.10,  # h
            0.60,  # j
            0.65,  # q
            1.00,  # x
            0.10,  # zh
            0.10,  # ch
            0.10,  # sh
            0.10,  # r
            0.20,  # z
            0.20,  # c
            0.10,  # s
            0.15,  # NONE
        ],
        # zh
        [
            0.00,  # DEFAULT
            0.10,  # b
            0.10,  # p
            0.10,  # m
            0.10,  # f
            0.10,  # d
            0.10,  # t
            0.10,  # n
            0.10,  # l
            0.10,  # g
            0.10,  # k
            0.10,  # h
            0.20,  # j
            0.20,  # q
            0.10,  # x
            1.00,  # zh
            0.70,  # ch
            0.60,  # sh
            0.40,  # r
            0.40,  # z
            0.20,  # c
            0.10,  # s
            0.15,  # NONE
        ],
        # ch
        [
            0.00,  # DEFAULT
            0.10,  # b
            0.10,  # p
            0.10,  # m
            0.10,  # f
            0.10,  # d
            0.10,  # t
            0.10,  # n
            0.10,  # l
            0.10,  # g
            0.10,  # k
            0.10,  # h
            0.20,  # j
            0.20,  # q
            0.10,  # x
            0.70,  # zh
            1.00,  # ch
            0.60,  # sh
            0.40,  # r
            0.20,  # z
            0.40,  # c
            0.10,  # s
            0.15,  # NONE
        ],
        # sh
        [
            0.00,  # DEFAULT
            0.10,  # b
            0.10,  # p
            0.10,  # m
            0.20,  # f
            0.10,  # d
            0.10,  # t
            0.10,  # n
            0.10,  # l
            0.10,  # g
            0.10,  # k
            0.10,  # h
            0.20,  # j
            0.20,  # q
            0.10,  # x
            0.60,  # zh
            0.60,  # ch
            1.00,  # sh
            0.40,  # r
            0.20,  # z
            0.20,  # c
            0.40,  # s
            0.15,  # NONE
        ],
        # r
        [
            0.00,  # DEFAULT
            0.10,  # b
            0.10,  # p
            0.10,  # m
            0.20,  # f
            0.10,  # d
            0.10,  # t
            0.20,  # n
            0.20,  # l
            0.10,  # g
            0.10,  # k
            0.10,  # h
            0.10,  # j
            0.10,  # q
            0.10,  # x
            0.40,  # zh
            0.40,  # ch
            0.40,  # sh
            1.00,  # r
            0.20,  # z
            0.20,  # c
            0.20,  # s
            0.15,  # NONE
        ],
        # z
        [
            0.00,  # DEFAULT
            0.20,  # b
            0.20,  # p
            0.10,  # m
            0.20,  # f
            0.20,  # d
            0.20,  # t
            0.20,  # n
            0.20,  # l
            0.10,  # g
            0.10,  # k
            0.20,  # h
            0.20,  # j
            0.20,  # q
            0.20,  # x
            0.40,  # zh
            0.20,  # ch
            0.20,  # sh
            0.20,  # r
            1.00,  # z
            0.70,  # c
            0.60,  # s
            0.15,  # NONE
        ],
        # c
        [
            0.00,  # DEFAULT
            0.20,  # b
            0.20,  # p
            0.10,  # m
            0.20,  # f
            0.20,  # d
            0.20,  # t
            0.20,  # n
            0.20,  # l
            0.10,  # g
            0.10,  # k
            0.20,  # h
            0.20,  # j
            0.20,  # q
            0.20,  # x
            0.20,  # zh
            0.40,  # ch
            0.20,  # sh
            0.20,  # r
            0.70,  # z
            1.00,  # c
            0.60,  # s
            0.15,  # NONE
        ],
        # s
        [
            0.00,  # DEFAULT
            0.10,  # b
            0.10,  # p
            0.10,  # m
            0.20,  # f
            0.10,  # d
            0.10,  # t
            0.10,  # n
            0.20,  # l
            0.10,  # g
            0.10,  # k
            0.20,  # h
            0.10,  # j
            0.10,  # q
            0.10,  # x
            0.10,  # zh
            0.10,  # ch
            0.40,  # sh
            0.20,  # r
            0.60,  # z
            0.60,  # c
            1.00,  # s
            0.15,  # NONE
        ],
        # NONE
        [
            0.00,  # DEFAULT
            0.15,  # b
            0.15,  # p
            0.15,  # m
            0.15,  # f
            0.15,  # d
            0.15,  # t
            0.15,  # n
            0.15,  # l
            0.15,  # g
            0.15,  # k
            0.15,  # h
            0.15,  # j
            0.15,  # q
            0.15,  # x
            0.15,  # zh
            0.15,  # ch
            0.15,  # sh
            0.15,  # r
            0.15,  # z
            0.15,  # c
            0.15,  # s
            1.00,  # NONE
        ],
    ],
    dtype=np.float16,
)

# Explanation of Matrix Structure:
# - Identical Finals (1.0): Cells where the same finals meet (e.g., A-A, U-U).
# - Very Close Finals (0.6-0.7): Finals that are very similar, differing only
#   slightly in articulation or nasalization (e.g., IA vs A, AN vs ANG).
# - Moderately Similar Finals (0.3-0.5): Finals that share some phonetic
#   features (e.g., O vs UO, AI vs EI).
# - Weak Similarity Finals (0.1-0.2): Finals with weak connections (e.g.,
#   U vs A, I vs O).
# - No Similarity (0.0): Finals with no shared features or completely different
#   sounds.
# - NONE Final: Assigns 0.15 to weak similarities with all other finals but 1.0
#   with itself.
SIM_MAT_FINAL = np.array(
    [
        # DEFAULT, I, U, V, A, IA, UA, O, UO, E, IE, VE, AI, UAI, EI, UEI, AO,
        # IAO, OU, IOU, AN, IAN, UAN, VAN, EN, IN, UEN, VN, ANG, IANG, UANG,
        # ENG, ING, UENG, ONG, IONG, ER, NONE,
        # DEFAULT
        [
            0.00,  # DEFAULT
            0.00,  # I
            0.00,  # U
            0.00,  # V
            0.00,  # A
            0.00,  # IA
            0.00,  # UA
            0.00,  # O
            0.00,  # UO
            0.00,  # E
            0.00,  # IE
            0.00,  # VE
            0.00,  # AI
            0.00,  # UAI
            0.00,  # EI
            0.00,  # UEI
            0.00,  # AO
            0.00,  # IAO
            0.00,  # OU
            0.00,  # IOU
            0.00,  # AN
            0.00,  # IAN
            0.00,  # UAN
            0.00,  # VAN
            0.00,  # EN
            0.00,  # IN
            0.00,  # UEN
            0.00,  # VN
            0.00,  # ANG
            0.00,  # IANG
            0.00,  # UANG
            0.00,  # ENG
            0.00,  # ING
            0.00,  # UENG
            0.00,  # ONG
            0.00,  # IONG
            0.00,  # ER
            0.00,  # NONE
        ],
        # I
        [
            0.00,  # DEFAULT
            1.00,  # I
            0.10,  # U
            0.10,  # V
            0.30,  # A
            0.50,  # IA
            0.30,  # UA
            0.20,  # O
            0.15,  # UO
            0.40,  # E
            0.80,  # IE
            0.30,  # VE
            0.30,  # AI
            0.20,  # UAI
            0.50,  # EI
            0.30,  # UEI
            0.10,  # AO
            0.30,  # IAO
            0.20,  # OU
            0.30,  # IOU
            0.10,  # AN
            0.30,  # IAN
            0.10,  # UAN
            0.20,  # VAN
            0.30,  # EN
            0.70,  # IN
            0.30,  # UEN
            0.30,  # VN
            0.10,  # ANG
            0.30,  # IANG
            0.20,  # UANG
            0.10,  # ENG
            0.60,  # ING
            0.10,  # UENG
            0.30,  # ONG
            0.10,  # IONG
            0.20,  # ER
            0.15,  # NONE
        ],
        # U
        [
            0.00,  # DEFAULT
            0.10,  # I
            1.00,  # U
            0.50,  # V
            0.30,  # A
            0.10,  # IA
            0.70,  # UA
            0.40,  # O
            0.80,  # UO
            0.20,  # E
            0.10,  # IE
            0.20,  # VE
            0.20,  # AI
            0.50,  # UAI
            0.30,  # EI
            0.70,  # UEI
            0.40,  # AO
            0.20,  # IAO
            0.80,  # OU
            0.60,  # IOU
            0.20,  # AN
            0.10,  # IAN
            0.60,  # UAN
            0.40,  # VAN
            0.30,  # EN
            0.20,  # IN
            0.60,  # UEN
            0.30,  # VN
            0.20,  # ANG
            0.10,  # IANG
            0.70,  # UANG
            0.20,  # ENG
            0.20,  # ING
            0.30,  # UENG
            0.70,  # ONG
            0.30,  # IONG
            0.20,  # ER
            0.15,  # NONE
        ],
        # V
        [
            0.00,  # DEFAULT
            0.10,  # I
            0.50,  # U
            1.00,  # V
            0.20,  # A
            0.10,  # IA
            0.20,  # UA
            0.10,  # O
            0.40,  # UO
            0.30,  # E
            0.20,  # IE
            0.60,  # VE
            0.20,  # AI
            0.10,  # UAI
            0.20,  # EI
            0.50,  # UEI
            0.10,  # AO
            0.20,  # IAO
            0.10,  # OU
            0.10,  # IOU
            0.20,  # AN
            0.30,  # IAN
            0.10,  # UAN
            0.60,  # VAN
            0.10,  # EN
            0.30,  # IN
            0.10,  # UEN
            0.70,  # VN
            0.10,  # ANG
            0.20,  # IANG
            0.10,  # UANG
            0.10,  # ENG
            0.10,  # ING
            0.10,  # UENG
            0.10,  # ONG
            0.10,  # IONG
            0.20,  # ER
            0.15,  # NONE
        ],
        # A
        [
            0.00,  # DEFAULT
            0.30,  # I
            0.30,  # U
            0.20,  # V
            1.00,  # A
            0.70,  # IA
            0.60,  # UA
            0.40,  # O
            0.30,  # UO
            0.50,  # E
            0.40,  # IE
            0.20,  # VE
            0.60,  # AI
            0.50,  # UAI
            0.20,  # EI
            0.30,  # UEI
            0.80,  # AO
            0.70,  # IAO
            0.30,  # OU
            0.40,  # IOU
            0.50,  # AN
            0.60,  # IAN
            0.50,  # UAN
            0.30,  # VAN
            0.50,  # EN
            0.30,  # IN
            0.30,  # UEN
            0.20,  # VN
            0.50,  # ANG
            0.70,  # IANG
            0.60,  # UANG
            0.30,  # ENG
            0.50,  # ING
            0.40,  # UENG
            0.30,  # ONG
            0.40,  # IONG
            0.40,  # ER
            0.15,  # NONE
        ],
        # IA
        [
            0.00,  # DEFAULT
            0.50,  # I
            0.10,  # U
            0.10,  # V
            0.70,  # A
            1.00,  # IA
            0.40,  # UA
            0.30,  # O
            0.20,  # UO
            0.60,  # E
            0.80,  # IE
            0.30,  # VE
            0.50,  # AI
            0.60,  # UAI
            0.20,  # EI
            0.30,  # UEI
            0.60,  # AO
            0.80,  # IAO
            0.20,  # OU
            0.30,  # IOU
            0.60,  # AN
            0.70,  # IAN
            0.50,  # UAN
            0.20,  # VAN
            0.60,  # EN
            0.40,  # IN
            0.30,  # UEN
            0.30,  # VN
            0.60,  # ANG
            0.80,  # IANG
            0.70,  # UANG
            0.30,  # ENG
            0.60,  # ING
            0.50,  # UENG
            0.30,  # ONG
            0.40,  # IONG
            0.40,  # ER
            0.15,  # NONE
        ],
        # UA
        [
            0.00,  # DEFAULT
            0.30,  # I
            0.70,  # U
            0.20,  # V
            0.60,  # A
            0.40,  # IA
            1.00,  # UA
            0.80,  # O
            0.40,  # UO
            0.40,  # E
            0.20,  # IE
            0.10,  # VE
            0.40,  # AI
            0.70,  # UAI
            0.40,  # EI
            0.60,  # UEI
            0.50,  # AO
            0.40,  # IAO
            0.80,  # OU
            0.50,  # IOU
            0.30,  # AN
            0.20,  # IAN
            0.70,  # UAN
            0.40,  # VAN
            0.20,  # EN
            0.30,  # IN
            0.50,  # UEN
            0.20,  # VN
            0.30,  # ANG
            0.40,  # IANG
            0.70,  # UANG
            0.40,  # ENG
            0.30,  # ING
            0.50,  # UENG
            0.80,  # ONG
            0.50,  # IONG
            0.30,  # ER
            0.15,  # NONE
        ],
        # O
        [
            0.00,  # DEFAULT
            0.20,  # I
            0.40,  # U
            0.10,  # V
            0.40,  # A
            0.30,  # IA
            0.80,  # UA
            1.00,  # O
            0.60,  # UO
            0.30,  # E
            0.10,  # IE
            0.20,  # VE
            0.20,  # AI
            0.40,  # UAI
            0.30,  # EI
            0.60,  # UEI
            0.50,  # AO
            0.30,  # IAO
            0.60,  # OU
            0.80,  # IOU
            0.10,  # AN
            0.20,  # IAN
            0.60,  # UAN
            0.40,  # VAN
            0.30,  # EN
            0.10,  # IN
            0.40,  # UEN
            0.20,  # VN
            0.10,  # ANG
            0.20,  # IANG
            0.80,  # UANG
            0.30,  # ENG
            0.40,  # ING
            0.20,  # UENG
            0.60,  # ONG
            0.40,  # IONG
            0.30,  # ER
            0.15,  # NONE
        ],
        # UO
        [
            0.00,  # DEFAULT
            0.15,  # I
            0.80,  # U
            0.40,  # V
            0.30,  # A
            0.20,  # IA
            0.40,  # UA
            0.60,  # O
            1.00,  # UO
            0.10,  # E
            0.15,  # IE
            0.20,  # VE
            0.20,  # AI
            0.60,  # UAI
            0.50,  # EI
            0.80,  # UEI
            0.40,  # AO
            0.20,  # IAO
            0.60,  # OU
            0.80,  # IOU
            0.30,  # AN
            0.10,  # IAN
            0.80,  # UAN
            0.40,  # VAN
            0.10,  # EN
            0.20,  # IN
            0.50,  # UEN
            0.30,  # VN
            0.10,  # ANG
            0.20,  # IANG
            0.60,  # UANG
            0.10,  # ENG
            0.40,  # ING
            0.50,  # UENG
            0.60,  # ONG
            0.80,  # IONG
            0.20,  # ER
            0.15,  # NONE
        ],
        # E
        [
            0.00,  # DEFAULT
            0.40,  # I
            0.20,  # U
            0.30,  # V
            0.50,  # A
            0.60,  # IA
            0.40,  # UA
            0.30,  # O
            0.10,  # UO
            1.00,  # E
            0.40,  # IE
            0.30,  # VE
            0.30,  # AI
            0.50,  # UAI
            0.80,  # EI
            0.40,  # UEI
            0.20,  # AO
            0.30,  # IAO
            0.20,  # OU
            0.30,  # IOU
            0.40,  # AN
            0.60,  # IAN
            0.50,  # UAN
            0.40,  # VAN
            0.80,  # EN
            0.30,  # IN
            0.50,  # UEN
            0.30,  # VN
            0.30,  # ANG
            0.50,  # IANG
            0.40,  # UANG
            0.40,  # ENG
            0.20,  # ING
            0.60,  # UENG
            0.40,  # ONG
            0.30,  # IONG
            0.40,  # ER
            0.15,  # NONE
        ],
        # IE
        [
            0.00,  # DEFAULT
            0.80,  # I
            0.10,  # U
            0.20,  # V
            0.40,  # A
            0.80,  # IA
            0.20,  # UA
            0.10,  # O
            0.15,  # UO
            0.40,  # E
            1.00,  # IE
            0.50,  # VE
            0.30,  # AI
            0.20,  # UAI
            0.80,  # EI
            0.50,  # UEI
            0.20,  # AO
            0.60,  # IAO
            0.15,  # OU
            0.40,  # IOU
            0.60,  # AN
            0.80,  # IAN
            0.30,  # UAN
            0.20,  # VAN
            0.50,  # EN
            0.30,  # IN
            0.30,  # UEN
            0.30,  # VN
            0.50,  # ANG
            0.60,  # IANG
            0.50,  # UANG
            0.20,  # ENG
            0.50,  # ING
            0.60,  # UENG
            0.50,  # ONG
            0.30,  # IONG
            0.40,  # ER
            0.15,  # NONE
        ],
        # VE
        [
            0.00,  # DEFAULT
            0.30,  # I
            0.20,  # U
            0.60,  # V
            0.20,  # A
            0.30,  # IA
            0.10,  # UA
            0.20,  # O
            0.20,  # UO
            0.30,  # E
            0.50,  # IE
            1.00,  # VE
            0.20,  # AI
            0.10,  # UAI
            0.20,  # EI
            0.60,  # UEI
            0.20,  # AO
            0.30,  # IAO
            0.20,  # OU
            0.10,  # IOU
            0.30,  # AN
            0.60,  # IAN
            0.40,  # UAN
            0.60,  # VAN
            0.40,  # EN
            0.50,  # IN
            0.60,  # UEN
            0.80,  # VN
            0.10,  # ANG
            0.30,  # IANG
            0.40,  # UANG
            0.30,  # ENG
            0.20,  # ING
            0.40,  # UENG
            0.20,  # ONG
            0.40,  # IONG
            0.20,  # ER
            0.15,  # NONE
        ],
        # AI
        [
            0.00,  # DEFAULT
            0.30,  # I
            0.20,  # U
            0.20,  # V
            0.60,  # A
            0.50,  # IA
            0.40,  # UA
            0.20,  # O
            0.20,  # UO
            0.30,  # E
            0.30,  # IE
            0.20,  # VE
            1.00,  # AI
            0.60,  # UAI
            0.30,  # EI
            0.50,  # UEI
            0.80,  # AO
            0.40,  # IAO
            0.60,  # OU
            0.50,  # IOU
            0.30,  # AN
            0.60,  # IAN
            0.50,  # UAN
            0.40,  # VAN
            0.50,  # EN
            0.30,  # IN
            0.60,  # UEN
            0.30,  # VN
            0.50,  # ANG
            0.60,  # IANG
            0.40,  # UANG
            0.30,  # ENG
            0.50,  # ING
            0.40,  # UENG
            0.50,  # ONG
            0.60,  # IONG
            0.40,  # ER
            0.15,  # NONE
        ],
        # UAI
        [
            0.00,  # DEFAULT
            0.20,  # I
            0.50,  # U
            0.10,  # V
            0.50,  # A
            0.60,  # IA
            0.70,  # UA
            0.40,  # O
            0.60,  # UO
            0.50,  # E
            0.20,  # IE
            0.10,  # VE
            0.60,  # AI
            1.00,  # UAI
            0.60,  # EI
            0.80,  # UEI
            0.50,  # AO
            0.20,  # IAO
            0.80,  # OU
            0.60,  # IOU
            0.30,  # AN
            0.60,  # IAN
            0.80,  # UAN
            0.70,  # VAN
            0.20,  # EN
            0.30,  # IN
            0.70,  # UEN
            0.20,  # VN
            0.60,  # ANG
            0.50,  # IANG
            0.80,  # UANG
            0.20,  # ENG
            0.50,  # ING
            0.40,  # UENG
            0.60,  # ONG
            0.50,  # IONG
            0.30,  # ER
            0.15,  # NONE
        ],
        # EI
        [
            0.00,  # DEFAULT
            0.50,  # I
            0.30,  # U
            0.20,  # V
            0.20,  # A
            0.20,  # IA
            0.40,  # UA
            0.30,  # O
            0.50,  # UO
            0.80,  # E
            0.80,  # IE
            0.20,  # VE
            0.30,  # AI
            0.60,  # UAI
            1.00,  # EI
            0.40,  # UEI
            0.20,  # AO
            0.60,  # IAO
            0.50,  # OU
            0.60,  # IOU
            0.20,  # AN
            0.30,  # IAN
            0.50,  # UAN
            0.60,  # VAN
            0.80,  # EN
            0.50,  # IN
            0.60,  # UEN
            0.50,  # VN
            0.30,  # ANG
            0.40,  # IANG
            0.50,  # UANG
            0.30,  # ENG
            0.50,  # ING
            0.60,  # UENG
            0.50,  # ONG
            0.30,  # IONG
            0.40,  # ER
            0.15,  # NONE
        ],
        # UEI
        [
            0.00,  # DEFAULT
            0.30,  # I
            0.70,  # U
            0.50,  # V
            0.30,  # A
            0.30,  # IA
            0.60,  # UA
            0.60,  # O
            0.80,  # UO
            0.40,  # E
            0.50,  # IE
            0.60,  # VE
            0.50,  # AI
            0.80,  # UAI
            0.40,  # EI
            1.00,  # UEI
            0.50,  # AO
            0.40,  # IAO
            0.80,  # OU
            0.60,  # IOU
            0.20,  # AN
            0.40,  # IAN
            0.60,  # UAN
            0.80,  # VAN
            0.50,  # EN
            0.40,  # IN
            0.70,  # UEN
            0.30,  # VN
            0.60,  # ANG
            0.50,  # IANG
            0.80,  # UANG
            0.50,  # ENG
            0.50,  # ING
            0.60,  # UENG
            0.80,  # ONG
            0.60,  # IONG
            0.40,  # ER
            0.15,  # NONE
        ],
        # AO
        [
            0.00,  # DEFAULT
            0.10,  # I
            0.40,  # U
            0.10,  # V
            0.80,  # A
            0.60,  # IA
            0.50,  # UA
            0.50,  # O
            0.40,  # UO
            0.20,  # E
            0.20,  # IE
            0.20,  # VE
            0.80,  # AI
            0.50,  # UAI
            0.20,  # EI
            0.50,  # UEI
            1.00,  # AO
            0.60,  # IAO
            0.50,  # OU
            0.60,  # IOU
            0.50,  # AN
            0.80,  # IAN
            0.50,  # UAN
            0.20,  # VAN
            0.50,  # EN
            0.40,  # IN
            0.20,  # UEN
            0.30,  # VN
            0.50,  # ANG
            0.70,  # IANG
            0.80,  # UANG
            0.40,  # ENG
            0.50,  # ING
            0.60,  # UENG
            0.50,  # ONG
            0.80,  # IONG
            0.30,  # ER
            0.15,  # NONE
        ],
        # IAO
        [
            0.00,  # DEFAULT
            0.30,  # I
            0.20,  # U
            0.20,  # V
            0.70,  # A
            0.80,  # IA
            0.40,  # UA
            0.30,  # O
            0.20,  # UO
            0.30,  # E
            0.60,  # IE
            0.30,  # VE
            0.40,  # AI
            0.20,  # UAI
            0.60,  # EI
            0.40,  # UEI
            0.60,  # AO
            1.00,  # IAO
            0.20,  # OU
            0.60,  # IOU
            0.30,  # AN
            0.60,  # IAN
            0.50,  # UAN
            0.30,  # VAN
            0.60,  # EN
            0.50,  # IN
            0.50,  # UEN
            0.40,  # VN
            0.50,  # ANG
            0.70,  # IANG
            0.40,  # UANG
            0.50,  # ENG
            0.50,  # ING
            0.60,  # UENG
            0.50,  # ONG
            0.40,  # IONG
            0.30,  # ER
            0.15,  # NONE
        ],
        # OU
        [
            0.00,  # DEFAULT
            0.20,  # I
            0.80,  # U
            0.10,  # V
            0.30,  # A
            0.20,  # IA
            0.80,  # UA
            0.60,  # O
            0.60,  # UO
            0.20,  # E
            0.15,  # IE
            0.20,  # VE
            0.60,  # AI
            0.80,  # UAI
            0.50,  # EI
            0.80,  # UEI
            0.50,  # AO
            0.20,  # IAO
            1.00,  # OU
            0.80,  # IOU
            0.20,  # AN
            0.50,  # IAN
            0.70,  # UAN
            0.50,  # VAN
            0.40,  # EN
            0.30,  # IN
            0.80,  # UEN
            0.50,  # VN
            0.50,  # ANG
            0.30,  # IANG
            0.60,  # UANG
            0.20,  # ENG
            0.40,  # ING
            0.30,  # UENG
            0.80,  # ONG
            0.50,  # IONG
            0.30,  # ER
            0.15,  # NONE
        ],
        # IOU
        [
            0.00,  # DEFAULT
            0.30,  # I
            0.60,  # U
            0.10,  # V
            0.40,  # A
            0.30,  # IA
            0.50,  # UA
            0.80,  # O
            0.80,  # UO
            0.30,  # E
            0.40,  # IE
            0.10,  # VE
            0.50,  # AI
            0.60,  # UAI
            0.60,  # EI
            0.60,  # UEI
            0.60,  # AO
            0.60,  # IAO
            0.80,  # OU
            1.00,  # IOU
            0.30,  # AN
            0.60,  # IAN
            0.50,  # UAN
            0.20,  # VAN
            0.50,  # EN
            0.60,  # IN
            0.80,  # UEN
            0.50,  # VN
            0.30,  # ANG
            0.50,  # IANG
            0.60,  # UANG
            0.30,  # ENG
            0.50,  # ING
            0.60,  # UENG
            0.80,  # ONG
            0.60,  # IONG
            0.30,  # ER
            0.15,  # NONE
        ],
        # AN
        [
            0.00,  # DEFAULT
            0.10,  # I
            0.20,  # U
            0.20,  # V
            0.50,  # A
            0.60,  # IA
            0.30,  # UA
            0.10,  # O
            0.30,  # UO
            0.40,  # E
            0.60,  # IE
            0.30,  # VE
            0.30,  # AI
            0.30,  # UAI
            0.20,  # EI
            0.20,  # UEI
            0.50,  # AO
            0.30,  # IAO
            0.20,  # OU
            0.30,  # IOU
            1.00,  # AN
            0.60,  # IAN
            0.50,  # UAN
            0.20,  # VAN
            0.50,  # EN
            0.50,  # IN
            0.60,  # UEN
            0.30,  # VN
            0.80,  # ANG
            0.50,  # IANG
            0.60,  # UANG
            0.20,  # ENG
            0.50,  # ING
            0.60,  # UENG
            0.50,  # ONG
            0.40,  # IONG
            0.30,  # ER
            0.15,  # NONE
        ],
        # IAN
        [
            0.00,  # DEFAULT
            0.30,  # I
            0.10,  # U
            0.30,  # V
            0.60,  # A
            0.80,  # IA
            0.40,  # UA
            0.20,  # O
            0.10,  # UO
            0.50,  # E
            0.70,  # IE
            0.40,  # VE
            0.30,  # AI
            0.30,  # UAI
            0.40,  # EI
            0.30,  # UEI
            0.40,  # AO
            0.50,  # IAO
            0.30,  # OU
            0.60,  # IOU
            0.60,  # AN
            1.00,  # IAN
            0.60,  # UAN
            0.30,  # VAN
            0.60,  # EN
            0.70,  # IN
            0.50,  # UEN
            0.60,  # VN
            0.50,  # ANG
            0.80,  # IANG
            0.70,  # UANG
            0.50,  # ENG
            0.60,  # ING
            0.50,  # UENG
            0.70,  # ONG
            0.50,  # IONG
            0.30,  # ER
            0.15,  # NONE
        ],
        # UAN
        [
            0.00,  # DEFAULT
            0.10,  # I
            0.60,  # U
            0.10,  # V
            0.50,  # A
            0.50,  # IA
            0.80,  # UA
            0.40,  # O
            0.60,  # UO
            0.30,  # E
            0.30,  # IE
            0.20,  # VE
            0.40,  # AI
            0.70,  # UAI
            0.50,  # EI
            0.60,  # UEI
            0.50,  # AO
            0.30,  # IAO
            0.70,  # OU
            0.50,  # IOU
            0.50,  # AN
            0.60,  # IAN
            1.00,  # UAN
            0.60,  # VAN
            0.40,  # EN
            0.30,  # IN
            0.70,  # UEN
            0.30,  # VN
            0.60,  # ANG
            0.50,  # IANG
            0.80,  # UANG
            0.30,  # ENG
            0.50,  # ING
            0.40,  # UENG
            0.70,  # ONG
            0.50,  # IONG
            0.30,  # ER
            0.15,  # NONE
        ],
        # VAN
        [
            0.00,  # DEFAULT
            0.20,  # I
            0.40,  # U
            0.60,  # V
            0.30,  # A
            0.20,  # IA
            0.60,  # UA
            0.40,  # O
            0.30,  # UO
            0.50,  # E
            0.30,  # IE
            0.70,  # VE
            0.20,  # AI
            0.10,  # UAI
            0.40,  # EI
            0.20,  # UEI
            0.20,  # AO
            0.10,  # IAO
            0.30,  # OU
            0.20,  # IOU
            0.20,  # AN
            0.30,  # IAN
            0.60,  # UAN
            1.00,  # VAN
            0.50,  # EN
            0.40,  # IN
            0.30,  # UEN
            0.80,  # VN
            0.30,  # ANG
            0.20,  # IANG
            0.30,  # UANG
            0.40,  # ENG
            0.20,  # ING
            0.40,  # UENG
            0.50,  # ONG
            0.30,  # IONG
            0.20,  # ER
            0.15,  # NONE
        ],
        # EN
        [
            0.00,  # DEFAULT
            0.30,  # I
            0.30,  # U
            0.10,  # V
            0.50,  # A
            0.60,  # IA
            0.30,  # UA
            0.20,  # O
            0.10,  # UO
            0.80,  # E
            0.50,  # IE
            0.40,  # VE
            0.30,  # AI
            0.30,  # UAI
            0.50,  # EI
            0.40,  # UEI
            0.30,  # AO
            0.50,  # IAO
            0.40,  # OU
            0.30,  # IOU
            0.50,  # AN
            0.60,  # IAN
            0.40,  # UAN
            0.50,  # VAN
            1.00,  # EN
            0.70,  # IN
            0.60,  # UEN
            0.40,  # VN
            0.50,  # ANG
            0.60,  # IANG
            0.40,  # UANG
            0.80,  # ENG
            0.50,  # ING
            0.50,  # UENG
            0.60,  # ONG
            0.50,  # IONG
            0.30,  # ER
            0.15,  # NONE
        ],
        # IN
        [
            0.00,  # DEFAULT
            0.70,  # I
            0.20,  # U
            0.30,  # V
            0.30,  # A
            0.40,  # IA
            0.20,  # UA
            0.10,  # O
            0.20,  # UO
            0.30,  # E
            0.30,  # IE
            0.50,  # VE
            0.30,  # AI
            0.30,  # UAI
            0.50,  # EI
            0.40,  # UEI
            0.40,  # AO
            0.50,  # IAO
            0.30,  # OU
            0.60,  # IOU
            0.50,  # AN
            0.70,  # IAN
            0.30,  # UAN
            0.40,  # VAN
            0.70,  # EN
            1.00,  # IN
            0.50,  # UEN
            0.30,  # VN
            0.50,  # ANG
            0.70,  # IANG
            0.50,  # UANG
            0.60,  # ENG
            0.70,  # ING
            0.50,  # UENG
            0.60,  # ONG
            0.70,  # IONG
            0.50,  # ER
            0.15,  # NONE
        ],
        # UEN
        [
            0.00,  # DEFAULT
            0.30,  # I
            0.60,  # U
            0.10,  # V
            0.30,  # A
            0.30,  # IA
            0.50,  # UA
            0.40,  # O
            0.50,  # UO
            0.30,  # E
            0.30,  # IE
            0.60,  # VE
            0.30,  # AI
            0.30,  # UAI
            0.60,  # EI
            0.70,  # UEI
            0.40,  # AO
            0.50,  # IAO
            0.80,  # OU
            0.60,  # IOU
            0.60,  # AN
            0.50,  # IAN
            0.70,  # UAN
            0.30,  # VAN
            0.60,  # EN
            0.50,  # IN
            1.00,  # UEN
            0.30,  # VN
            0.50,  # ANG
            0.40,  # IANG
            0.60,  # UANG
            0.50,  # ENG
            0.40,  # ING
            0.50,  # UENG
            0.80,  # ONG
            0.60,  # IONG
            0.50,  # ER
            0.15,  # NONE
        ],
        # VN
        [
            0.00,  # DEFAULT
            0.30,  # I
            0.30,  # U
            0.70,  # V
            0.20,  # A
            0.30,  # IA
            0.30,  # UA
            0.20,  # O
            0.30,  # UO
            0.40,  # E
            0.30,  # IE
            0.80,  # VE
            0.30,  # AI
            0.30,  # UAI
            0.40,  # EI
            0.30,  # UEI
            0.20,  # AO
            0.30,  # IAO
            0.30,  # OU
            0.50,  # IOU
            0.30,  # AN
            0.60,  # IAN
            0.30,  # UAN
            0.80,  # VAN
            0.40,  # EN
            0.50,  # IN
            0.30,  # UEN
            1.00,  # VN
            0.30,  # ANG
            0.50,  # IANG
            0.60,  # UANG
            0.30,  # ENG
            0.40,  # ING
            0.50,  # UENG
            0.30,  # ONG
            0.60,  # IONG
            0.40,  # ER
            0.15,  # NONE
        ],
        # ANG
        [
            0.00,  # DEFAULT
            0.10,  # I
            0.20,  # U
            0.10,  # V
            0.80,  # A
            0.60,  # IA
            0.60,  # UA
            0.10,  # O
            0.10,  # UO
            0.30,  # E
            0.50,  # IE
            0.10,  # VE
            0.50,  # AI
            0.60,  # UAI
            0.30,  # EI
            0.60,  # UEI
            0.50,  # AO
            0.50,  # IAO
            0.50,  # OU
            0.30,  # IOU
            0.80,  # AN
            0.50,  # IAN
            0.60,  # UAN
            0.30,  # VAN
            0.50,  # EN
            0.50,  # IN
            0.50,  # UEN
            0.30,  # VN
            1.00,  # ANG
            0.70,  # IANG
            0.60,  # UANG
            0.60,  # ENG
            0.50,  # ING
            0.70,  # UENG
            0.50,  # ONG
            0.60,  # IONG
            0.30,  # ER
            0.15,  # NONE
        ],
        # IANG
        [
            0.00,  # DEFAULT
            0.30,  # I
            0.10,  # U
            0.20,  # V
            0.70,  # A
            0.80,  # IA
            0.50,  # UA
            0.20,  # O
            0.30,  # UO
            0.50,  # E
            0.60,  # IE
            0.30,  # VE
            0.50,  # AI
            0.60,  # UAI
            0.40,  # EI
            0.50,  # UEI
            0.70,  # AO
            0.70,  # IAO
            0.30,  # OU
            0.50,  # IOU
            0.50,  # AN
            0.80,  # IAN
            0.50,  # UAN
            0.20,  # VAN
            0.60,  # EN
            0.70,  # IN
            0.40,  # UEN
            0.50,  # VN
            0.70,  # ANG
            1.00,  # IANG
            0.80,  # UANG
            0.50,  # ENG
            0.60,  # ING
            0.70,  # UENG
            0.60,  # ONG
            0.50,  # IONG
            0.50,  # ER
            0.15,  # NONE
        ],
        # UANG
        [
            0.00,  # DEFAULT
            0.20,  # I
            0.70,  # U
            0.10,  # V
            0.60,  # A
            0.70,  # IA
            0.80,  # UA
            0.40,  # O
            0.60,  # UO
            0.40,  # E
            0.50,  # IE
            0.30,  # VE
            0.60,  # AI
            0.80,  # UAI
            0.50,  # EI
            0.80,  # UEI
            0.80,  # AO
            0.70,  # IAO
            0.60,  # OU
            0.50,  # IOU
            0.60,  # AN
            0.70,  # IAN
            0.80,  # UAN
            0.60,  # VAN
            0.40,  # EN
            0.50,  # IN
            0.60,  # UEN
            0.60,  # VN
            0.60,  # ANG
            0.80,  # IANG
            1.00,  # UANG
            0.50,  # ENG
            0.60,  # ING
            0.70,  # UENG
            0.80,  # ONG
            0.60,  # IONG
            0.50,  # ER
            0.15,  # NONE
        ],
        # ENG
        [
            0.00,  # DEFAULT
            0.10,  # I
            0.20,  # U
            0.10,  # V
            0.30,  # A
            0.30,  # IA
            0.30,  # UA
            0.20,  # O
            0.10,  # UO
            0.80,  # E
            0.40,  # IE
            0.30,  # VE
            0.30,  # AI
            0.40,  # UAI
            0.30,  # EI
            0.40,  # UEI
            0.40,  # AO
            0.50,  # IAO
            0.30,  # OU
            0.30,  # IOU
            0.50,  # AN
            0.50,  # IAN
            0.40,  # UAN
            0.30,  # VAN
            0.80,  # EN
            0.50,  # IN
            0.50,  # UEN
            0.30,  # VN
            0.60,  # ANG
            0.50,  # IANG
            0.50,  # UANG
            1.00,  # ENG
            0.60,  # ING
            0.60,  # UENG
            0.50,  # ONG
            0.70,  # IONG
            0.30,  # ER
            0.15,  # NONE
        ],
        # ING
        [
            0.00,  # DEFAULT
            0.60,  # I
            0.20,  # U
            0.10,  # V
            0.50,  # A
            0.60,  # IA
            0.20,  # UA
            0.30,  # O
            0.40,  # UO
            0.50,  # E
            0.50,  # IE
            0.20,  # VE
            0.50,  # AI
            0.50,  # UAI
            0.30,  # EI
            0.50,  # UEI
            0.50,  # AO
            0.50,  # IAO
            0.50,  # OU
            0.30,  # IOU
            0.50,  # AN
            0.70,  # IAN
            0.50,  # UAN
            0.20,  # VAN
            0.50,  # EN
            0.60,  # IN
            0.40,  # UEN
            0.50,  # VN
            0.50,  # ANG
            0.60,  # IANG
            0.60,  # UANG
            0.60,  # ENG
            1.00,  # ING
            0.50,  # UENG
            0.60,  # ONG
            0.50,  # IONG
            0.40,  # ER
            0.15,  # NONE
        ],
        # UENG
        [
            0.00,  # DEFAULT
            0.10,  # I
            0.30,  # U
            0.10,  # V
            0.40,  # A
            0.50,  # IA
            0.50,  # UA
            0.50,  # O
            0.50,  # UO
            0.50,  # E
            0.50,  # IE
            0.40,  # VE
            0.30,  # AI
            0.50,  # UAI
            0.30,  # EI
            0.60,  # UEI
            0.50,  # AO
            0.50,  # IAO
            0.80,  # OU
            0.60,  # IOU
            0.60,  # AN
            0.50,  # IAN
            0.70,  # UAN
            0.40,  # VAN
            0.60,  # EN
            0.50,  # IN
            0.80,  # UEN
            0.50,  # VN
            0.70,  # ANG
            0.60,  # IANG
            0.80,  # UANG
            0.60,  # ENG
            0.50,  # ING
            0.70,  # UENG
            1.00,  # ONG
            0.70,  # IONG
            0.50,  # ER
            0.15,  # NONE
        ],
        # ONG
        [
            0.00,  # DEFAULT
            0.30,  # I
            0.70,  # U
            0.10,  # V
            0.30,  # A
            0.30,  # IA
            0.80,  # UA
            0.60,  # O
            0.60,  # UO
            0.40,  # E
            0.50,  # IE
            0.30,  # VE
            0.30,  # AI
            0.50,  # UAI
            0.30,  # EI
            0.60,  # UEI
            0.50,  # AO
            0.50,  # IAO
            0.60,  # OU
            0.50,  # IOU
            0.50,  # AN
            0.50,  # IAN
            0.60,  # UAN
            0.30,  # VAN
            0.50,  # EN
            0.60,  # IN
            0.60,  # UEN
            0.50,  # VN
            0.50,  # ANG
            0.50,  # IANG
            0.80,  # UANG
            0.50,  # ENG
            0.60,  # ING
            0.70,  # UENG
            0.70,  # ONG
            1.00,  # IONG
            0.50,  # ER
            0.15,  # NONE
        ],
        # IONG
        [
            0.00,  # DEFAULT
            0.10,  # I
            0.30,  # U
            0.10,  # V
            0.40,  # A
            0.40,  # IA
            0.50,  # UA
            0.40,  # O
            0.50,  # UO
            0.30,  # E
            0.50,  # IE
            0.40,  # VE
            0.50,  # AI
            0.60,  # UAI
            0.30,  # EI
            0.50,  # UEI
            0.50,  # AO
            0.50,  # IAO
            0.50,  # OU
            0.60,  # IOU
            0.40,  # AN
            0.50,  # IAN
            0.50,  # UAN
            0.30,  # VAN
            0.50,  # EN
            0.50,  # IN
            0.60,  # UEN
            0.60,  # VN
            0.60,  # ANG
            0.50,  # IANG
            0.60,  # UANG
            0.50,  # ENG
            0.60,  # ING
            0.50,  # UENG
            0.70,  # ONG
            0.70,  # IONG
            1.00,  # ER
            0.15,  # NONE
        ],
        # ER
        [
            0.00,  # DEFAULT
            0.20,  # I
            0.20,  # U
            0.20,  # V
            0.40,  # A
            0.40,  # IA
            0.30,  # UA
            0.30,  # O
            0.20,  # UO
            0.40,  # E
            0.40,  # IE
            0.20,  # VE
            0.40,  # AI
            0.30,  # UAI
            0.40,  # EI
            0.40,  # UEI
            0.30,  # AO
            0.30,  # IAO
            0.30,  # OU
            0.30,  # IOU
            0.30,  # AN
            0.30,  # IAN
            0.30,  # UAN
            0.20,  # VAN
            0.30,  # EN
            0.50,  # IN
            0.50,  # UEN
            0.40,  # VN
            0.30,  # ANG
            0.50,  # IANG
            0.50,  # UANG
            0.70,  # ENG
            0.50,  # ING
            0.50,  # UENG
            0.50,  # ONG
            0.50,  # IONG
            0.50,  # ER
            1.00,  # NONE
        ],
        # NONE
        [
            0.00,  # DEFAULT
            0.15,  # I
            0.15,  # U
            0.15,  # V
            0.15,  # A
            0.15,  # IA
            0.15,  # UA
            0.15,  # O
            0.15,  # UO
            0.15,  # E
            0.15,  # IE
            0.15,  # VE
            0.15,  # AI
            0.15,  # UAI
            0.15,  # EI
            0.15,  # UEI
            0.15,  # AO
            0.15,  # IAO
            0.15,  # OU
            0.15,  # IOU
            0.15,  # AN
            0.15,  # IAN
            0.15,  # UAN
            0.15,  # VAN
            0.15,  # EN
            0.15,  # IN
            0.15,  # UEN
            0.15,  # VN
            0.15,  # ANG
            0.15,  # IANG
            0.15,  # UANG
            0.15,  # ENG
            0.15,  # ING
            0.15,  # UENG
            0.15,  # ONG
            0.15,  # IONG
            0.15,  # ER
            1.00,  # NONE
        ],
    ],
    dtype=np.float16,
)


SIM_MAT_TONE = np.array(
    [
        # DEFAULT
        [0, 0, 0, 0, 0, 0],
        # 1
        [0, 1, 0, 0, 0, 0],
        # 2
        [0, 0, 1, 0, 0, 0],
        # 3
        [0, 0, 0, 1, 0, 0],
        # 4
        [0, 0, 0, 0, 1, 0],
        # neutral
        [0, 0, 0, 0, 0, 1],
    ],
    dtype=np.float16,
)


def _get_init_ind(cchar: str) -> int:
    try:
        ch_init = lazy_pinyin(cchar, style=Style.INITIALS)[0]
        return _MAP_STR2INIT.get(ch_init, 0)
    except IndexError:
        return 0


def _get_final_ind(cchar: str) -> int:
    try:
        ch_final = lazy_pinyin(cchar, style=Style.FINALS)[0]
        return _MAP_STR2FINAL.get(ch_final, 0)
    except IndexError:
        return 0


def _get_tone_ind(cchar: str) -> int:
    try:
        ch_tone = lazy_pinyin(
            cchar,
            style=Style.TONE3,
            neutral_tone_with_five=True,
        )[0][-1]
        return _MAP_STR2TONE.get(ch_tone, 0)
    except IndexError:
        return 0


def _sim_init(init1: int, init2: int) -> float:
    return SIM_MAT_INITIAL[init1, init2]


def _sim_final(final1: int, final2: int) -> float:
    return SIM_MAT_FINAL[final1, final2]


def _sim_tone(tone1: int, tone2: int) -> float:
    return SIM_MAT_TONE[tone1, tone2]


def get_ind(cchar: str) -> tuple[int, int, int]:
    """Get the phonetic feature indices of a Chinese character.

    Args:
        cchar (str): one Chinese character.

    Returns:
        tuple[int, int, int]: The phonetic features of the Chinese character.
            Default to 0s if the character is not found.
    """
    try:
        ch_init = lazy_pinyin(cchar, style=Style.INITIALS)[0]
        ch_final = lazy_pinyin(cchar, style=Style.FINALS)[0]
        ch_tone = lazy_pinyin(
            cchar,
            style=Style.TONE3,
            neutral_tone_with_five=True,
        )[0][-1]
        init_int = _MAP_STR2INIT.get(ch_init, 0)
        final_int = _MAP_STR2FINAL.get(ch_final, 0)
        tone_int = _MAP_STR2TONE.get(ch_tone, 0)
    except IndexError:
        init_int, final_int, tone_int = 0, 0, 0
    return init_int, final_int, tone_int


def sim(cchar1: str, cchar2: str) -> float:
    """Calculate the phonetic similarity between two Chinese characters."""
    init1, final1, tone1 = get_ind(cchar1)
    init2, final2, tone2 = get_ind(cchar2)
    sim_init = SIM_MAT_INITIAL[init1, init2]
    sim_final = SIM_MAT_FINAL[final1, final2]
    sim_tone = SIM_MAT_TONE[tone1, tone2]
    return (
        _WEIGHT_INITIAL * sim_init
        + _WEIGHT_FINAL * sim_final
        + _WEIGHT_TONE * sim_tone
    )


# Vectorized function
def get_ind_vec(cchars: np.ndarray) -> np.ndarray:
    """Vectorized version of `get_ind`."""
    inits = np.vectorize(_get_init_ind, otypes=[np.uint8])(cchars)
    finals = np.vectorize(_get_final_ind, otypes=[np.uint8])(cchars)
    tones = np.vectorize(_get_tone_ind, otypes=[np.uint8])(cchars)
    return np.stack([inits, finals, tones], axis=1).astype(np.uint8)


def sim_vec(cchars1: np.ndarray, cchars2: np.ndarray) -> np.ndarray:
    """Vectorized version of `sim`."""
    ar_pinyin_1 = get_ind_vec(cchars1)
    ar_pinyin_2 = get_ind_vec(cchars2)
    sim_inits = np.vectorize(_sim_init, otypes=[np.float32])(
        ar_pinyin_1[:, 0], ar_pinyin_2[:, 0]
    )
    sim_finals = np.vectorize(_sim_final, otypes=[np.float32])(
        ar_pinyin_1[:, 1], ar_pinyin_2[:, 1]
    )
    sim_tones = np.vectorize(_sim_tone, otypes=[np.float32])(
        ar_pinyin_1[:, 2], ar_pinyin_2[:, 2]
    )
    return (
        _WEIGHT_INITIAL * sim_inits
        + _WEIGHT_FINAL * sim_finals
        + _WEIGHT_TONE * sim_tones
    )
