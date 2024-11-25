"""Parse SRT files."""

import re

import numpy as np

from subtitle_alchemy.utils import timeline

N_BLOCK_LINES = 3


def generate(srt_string: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse SRT string into arrays of lines and timestamps.

    Args:
        srt_string (str): Input SRT string.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of
        - lines: Array of subtitle text lines (N,)
        - timestamps: Array of start/end timestamps in seconds (N, 2)
    """
    # Split into subtitle blocks
    blocks = re.split(r"\n\n+", srt_string.strip())

    lines = []
    timestamps = []

    for block in blocks:
        if not block.strip():
            continue

        # Split block into lines
        block_lines = block.split("\n")
        if len(block_lines) < N_BLOCK_LINES:
            continue

        # Parse timestamp line (second line)
        tl_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> "
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})",
            block_lines[1],
        )
        if not tl_match:
            continue

        # Convert timestamp to milliseconds
        times = [int(x) for x in tl_match.groups()]
        ms_start = timeline.hmss2ms(*times[:4])
        ms_end = timeline.hmss2ms(*times[4:])

        # Join remaining lines as subtitle text
        subtitle_text = " ".join(block_lines[2:])

        lines.append(subtitle_text)
        timestamps.append([ms_start, ms_end])

    return np.array(lines, dtype="U"), np.array(timestamps, dtype=np.int64)
