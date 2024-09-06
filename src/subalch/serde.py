"""Serialization and deserialization of transcript data."""

from pathlib import Path

import numpy as np

_K_TXT = "text"
_K_TL = "timeline"

__all__ = ["save", "load"]


def save(key: str, folder: Path, txt: np.ndarray, tl: np.ndarray) -> None:
    """Save transcript as npz file.

    Args:
        key (str): file name
        folder (Path): folder to save the npz file
        txt (np.ndarray): 1D UTF8 encoded array (N, ) of space-separated words
        tl (np.ndarray): 2D int32 timeline array (N, 2) of start-end timestamp
            pairs
    """
    np.savez(folder / f"{key}.npz", text=txt, timeline=tl)


def load(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """Load transcript from npz file.

    Args:
        path (Path): npz file path

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of words and timeline arrays:
        - key: str file name
        - txt: UTF-8 encoded words array (N, )
        - ts: int32 timeline array (N, 2) i.e. start-end timestamp pairs
    """
    with np.load(path) as data:
        return path.stem, data[_K_TXT], data[_K_TL]
