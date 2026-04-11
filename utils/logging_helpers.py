"""Structured, coloured terminal logging for multi-agent nodes."""

from __future__ import annotations

import sys

from utils.constants import BOLD, RESET


def agent_log(agent: str, color: str, msg: str) -> None:
    """Print a clearly labelled, coloured log line to stderr."""
    header = f"{color}{BOLD}[{agent}]{RESET}"
    for line in msg.strip().splitlines():
        print(f"  {header} {line}", file=sys.stderr, flush=True)
    print(file=sys.stderr, flush=True)


def divider(title: str) -> None:
    """Print a section divider to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")
