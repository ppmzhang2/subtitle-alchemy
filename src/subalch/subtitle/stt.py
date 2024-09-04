"""STT Service."""

from pathlib import Path

from funasr import AutoModel
from loguru import logger
import numpy as np

_K_TXT = "txt"
_K_TL = "tl"

__all__ = ["save", "load"]


def save(model: str, audio: Path, folder: Path, hotword: str) -> None:
    """Generate and save transcript from audio file.

    The return value is expected as a list with a single dict in it.
    The dict should have the following keys:
    - key: file name
    - txt: space-separated words
    - ts: list of timestamp pair lists

    It saves the transcript as a npz file with the following keys:
    - txt: UTF-8 encoded words array (N, )
    - ts: int32 timeline array (N, 2) i.e. start-end timestamp pairs

    Args:
        model (str): Model name
        audio (Path): Path to audio file
        folder (Path): Directory to save transcript
        hotword (str): Hotword to detect
    """
    model = AutoModel(
        model=model,
        vad_model="fsmn-vad",
        log_level="ERROR",
    )
    res = model.generate(input=str(audio), batch_size_s=300, hotword=hotword)
    if not isinstance(res, list):
        logger.error(f"Expected list, got {type(res)}")
        raise TypeError()

    res = res[0]
    if not isinstance(res, dict):
        logger.error(f"Expected dict in list, got {type(res)}")
        raise TypeError()

    key = res["key"]
    arr_txt = np.array(res["text"].split(), dtype="U")
    arr_ts = np.array(res["timestamp"], dtype=np.int32)
    np.savez(folder / f"{key}.npz", **{_K_TXT: arr_txt, _K_TL: arr_ts})


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load transcript from npz file.

    Args:
        path (Path): npz file path

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of words and timeline arrays
    """
    with np.load(path) as data:
        return data[_K_TXT], data[_K_TL]
