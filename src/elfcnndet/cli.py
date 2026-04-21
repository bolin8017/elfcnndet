"""CLI entry point. Auto-generated from maldet spec."""

from maldet import create_cli

from .detector import ElfCnnDetector

app = create_cli(ElfCnnDetector)


if __name__ == "__main__":
    app()
