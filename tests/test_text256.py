"""Unit tests for text256 feature extraction (duplicated from elfrfdet)."""

from pathlib import Path

import numpy as np
import pytest

from elfcnndet.text256 import extract_text256


def _system_elf() -> Path:
    for candidate in ("/bin/ls", "/usr/bin/ls", "/bin/cat"):
        p = Path(candidate)
        if p.is_file():
            return p
    pytest.skip("no system ELF available")


def test_returns_uint8_vector_of_requested_size() -> None:
    vec = extract_text256(_system_elf(), size=256)
    assert vec.dtype == np.uint8
    assert vec.shape == (256,)


def test_zero_pads_short_sections() -> None:
    from elftools.elf.elffile import ELFFile
    elf_path = _system_elf()
    with open(elf_path, "rb") as f:
        text_len = len(ELFFile(f).get_section_by_name(".text").data())
    vec = extract_text256(elf_path, size=text_len + 1024)
    assert vec[text_len:].sum() == 0


def test_raises_on_non_elf(tmp_path: Path) -> None:
    p = tmp_path / "notelf"
    p.write_bytes(b"hello")
    with pytest.raises(ValueError, match="not a valid ELF"):
        extract_text256(p)
