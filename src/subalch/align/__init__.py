"""Align subtitles using GT transcript and ASR output."""

from subalch.align._index import get_aligned_index
from subalch.align._tl import get_aligned_tl

__all__ = [
    "get_aligned_index",
    "get_aligned_tl",
]
