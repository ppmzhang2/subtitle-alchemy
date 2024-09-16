"""Align subtitles using GT transcript and ASR output."""

from subtitle_alchemy.align._index import get_aligned_index
from subtitle_alchemy.align._tl import get_aligned_tl

__all__ = [
    "get_aligned_index",
    "get_aligned_tl",
]
