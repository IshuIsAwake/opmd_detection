"""
console.py — Loud, uniform run banners and key/value printing.

Every script frames its work with a START / END banner naming model + arm +
out_dir, so a long terminal scroll stays legible and the experiment arm is
never ambiguous. Pure stdout; no logging framework, no deps.
"""

from __future__ import annotations

from typing import Any

_WIDTH = 72


def phase(title: str) -> None:
    """A loud, impossible-to-miss section banner."""
    bar = "═" * _WIDTH
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def kv(pairs: dict[str, Any], indent: int = 2) -> None:
    """Aligned key: value block (skips None values)."""
    items = [(k, v) for k, v in pairs.items() if v is not None]
    if not items:
        return
    width = max(len(str(k)) for k, _ in items)
    pad = " " * indent
    for k, v in items:
        print(f"{pad}{str(k):<{width}} : {v}", flush=True)


def start(script: str, **fields: Any) -> None:
    """START banner for a step (e.g. start('Step 2 · train detector', arm=...))."""
    phase(f"▶ START — {script}")
    kv(fields)


def end(script: str, **fields: Any) -> None:
    """END banner for a step."""
    kv(fields)
    phase(f"■ END — {script}")
