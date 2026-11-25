"""Lock and verify all prior project files and source hashes."""
import argparse
from pathlib import Path

from if10_research import freeze
from if17_research.catalog import ref
from if17_research.freeze import verify
from p1.io import dump

from .protocol import HERE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--card", type=Path)
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    freeze.HERE = HERE
    if args.verify:
        verify(out)
    else:
        out.mkdir(exist_ok=False, parents=True)
        media, old = freeze.sources(freeze.MEDIA)
        prior = freeze.history(out)
        dump(out / "history_lock.json", {"history": prior, "historical_sources": old, "media": media,
                                         "card": ref(args.card.resolve())})
