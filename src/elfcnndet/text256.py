"""Extract the first 256 bytes of the `.text` section from an ELF file.

Identical to elfrfdet/text256.py — duplicated on purpose so each template
repo is self-contained and can be forked / modified independently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile

DEFAULT_SIZE = 256


def extract_text256(path: Path, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Return the first `size` bytes of `.text` as a uint8 vector.

    Args:
        path: Path to an ELF file.
        size: Number of bytes to extract (default 256).

    Returns:
        Numpy array of shape (size,), dtype uint8, zero-padded.

    Raises:
        ValueError: File is not a valid ELF or has no `.text` section.
        FileNotFoundError: File does not exist.
    """
    with open(path, "rb") as f:
        try:
            elf = ELFFile(f)
        except ELFError as exc:
            raise ValueError(f"{path.name}: not a valid ELF ({exc})") from exc

        text = elf.get_section_by_name(".text")
        if text is None:
            raise ValueError(f"{path.name}: no .text section")
        data = text.data()[:size]

    vec = np.zeros(size, dtype=np.uint8)
    vec[: len(data)] = np.frombuffer(data, dtype=np.uint8)
    return vec
