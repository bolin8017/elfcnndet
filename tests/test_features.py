"""Unit tests for elfcnndet.features.Text256Extractor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from elfcnndet.features import Text256Extractor
from maldet.types import Sample


def _system_elf() -> Path:
    for candidate in ("/bin/ls", "/usr/bin/ls", "/bin/cat", "/usr/bin/cat"):
        p = Path(candidate)
        if p.is_file():
            return p
    pytest.skip("no system ELF available")


def _sample(path: Path) -> Sample:
    return Sample(sha256="0" * 64, path=path, label="Benign")


def test_returns_uint8_vector_of_default_size() -> None:
    extractor = Text256Extractor()
    vec = extractor.extract(_sample(_system_elf()))
    assert vec.dtype == np.uint8
    assert vec.shape == (256,)


def test_zero_pads_short_text() -> None:
    extractor = Text256Extractor(size=8192, pad_value=0)
    vec = extractor.extract(_sample(_system_elf()))
    assert vec.shape == (8192,)
    # Some trailing bytes must be padding zeros (real .text rarely fills 8192).
    assert vec[-256:].sum() < 256 * 255


def test_truncated_elf_raises_value_error(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.elf"
    truncated.write_bytes(b"\x7fELF" + b"\x00" * 12 + b"\x99")
    extractor = Text256Extractor()
    with pytest.raises(ValueError, match="ELF parse failed"):
        extractor.extract(_sample(truncated))


def test_no_text_section_raises_value_error(tmp_path: Path) -> None:
    not_elf = tmp_path / "not_an_elf.bin"
    not_elf.write_bytes(b"this is not an ELF binary")
    extractor = Text256Extractor()
    with pytest.raises(ValueError, match="ELF parse failed"):
        extractor.extract(_sample(not_elf))
