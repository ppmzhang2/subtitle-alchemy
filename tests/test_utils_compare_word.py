"""Test pinyin similarity in utils."""

from subalch.utils.compare_word import pinyin_similarity


def test_pinyin_similarity() -> None:
    """Test `pinyin_similarity` function."""
    char1 = "妈"
    char2 = "马"
    sim_score = pinyin_similarity(char1, char2)
    sim_score_exp = 0.8
    assert sim_score == sim_score_exp
