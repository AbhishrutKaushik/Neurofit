"""Mess with strings in silly ways."""

from __future__ import annotations


def mash(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return f"{text.upper()} | {text[::-1]} | {text.title()}"


if __name__ == "__main__":
    sample = "neurofit slop mode"
    print(mash(sample))

