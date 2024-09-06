"""Compare the phonetic similarity between two Chinese characters."""

from __future__ import annotations

from dataclasses import dataclass

from pypinyin import Style
from pypinyin import lazy_pinyin


@dataclass
class PinYin:
    """Pinyin Phonetic Features."""

    initial: str
    final: str
    tone: str

    def __eq__(self, other: PinYin) -> bool:
        return (self.initial == other.initial and self.final == other.final
                and self.tone == other.tone)

    def __hash__(self):
        return hash((self.initial, self.final, self.tone))

    def __repr__(self):
        return f"{self.initial}{self.final}{self.tone}"

    def similarity(
            self,
            other: PinYin,
            weights: tuple[float] = (0.4, 0.4, 0.2),
    ) -> float:
        """Calculate the similarity between two `PinYin` objects."""
        sim_init = 1 if self.initial == other.initial else 0
        sim_final = 1 if self.final == other.final else 0
        sim_tone = 1 if self.tone == other.tone else 0

        return (sim_init * weights[0] + sim_final * weights[1] +
                sim_tone * weights[2])


def get_pinyin(cchar: str) -> PinYin:
    """Get the phonetic features `PinYin` of a Chinese character."""
    initial = lazy_pinyin(cchar, style=Style.INITIALS)[0]
    final = lazy_pinyin(cchar, style=Style.FINALS)[0]
    tone = lazy_pinyin(cchar, style=Style.TONE3)[0][-1]
    return PinYin(initial, final, tone)


def pinyin_similarity(cchar1: str, cchar2: str) -> float:
    """Calculate the phonetic similarity between two Chinese characters."""
    pinyin1 = get_pinyin(cchar1)
    pinyin2 = get_pinyin(cchar2)
    return pinyin1.similarity(pinyin2)
