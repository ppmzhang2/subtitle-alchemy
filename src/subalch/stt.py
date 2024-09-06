"""STT Service."""

from pathlib import Path

from funasr import AutoModel
from loguru import logger
import numpy as np

__all__ = ["generate"]


def generate(
    model: str, audio: Path, hotword: str = "",
) -> tuple[str, np.ndarray, np.ndarray]:
    """Generate transcript from audio file.

    The return value is expected as a list with a single dict in it.
    The dict should have the following keys:
    - key: file name
    - txt: space-separated words
    - ts: list of timestamp pair lists

    Args:
        model (str): Model name
        audio (Path): Path to audio file
        hotword (str): Hot words to detect, separated by space, default ""

    Returns:
        tuple[str, np.ndarray, np.ndarray]: Tuple of key, words and timeline
            arrays:
        - key: file name
        - words: 1D UTF8 encoded array (N, ) of space-separated words
        - timeline: 2D int32 timeline array (N, 2) of start-end timestamp pairs
    """
    model = AutoModel(
        model=model,
        vad_model="fsmn-vad",
        log_level="ERROR",
    )
    if hotword == "":
        res = model.generate(input=str(audio), batch_size_s=300)
    else:
        res = model.generate(
            input=str(audio), batch_size_s=300, hotword=hotword
        )
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
    return key, arr_txt, arr_ts
