"""Project-wide constants."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# ANSI colours for terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"
