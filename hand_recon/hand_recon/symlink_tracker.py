"""Context manager that snapshots and removes stale HaWoR symlinks.

HaWoR internal code writes outputs using paths like ``<cwd_root>/<seq>/...``
where ``<cwd_root>`` depends on the directory shape. To redirect outputs we
pre-place symlinks that bake in the destination.

This tracker records any symlinks it creates (or pre-existing ones it would
otherwise not disturb) and, on cleanup, walks every directory that previously
contained one of ours, deleting links whose targets now 404. That prevents
broken symlinks from poisoning later glob searches.
"""

from __future__ import annotations

import os
from pathlib import Path


class SymlinkTracker:
    """Tracks source→dest symlink pairs and cleans stale ones on exit."""

    def __init__(self) -> None:
        self._created: list[Path] = []
        self._watch_dirs: set[Path] = set()

    def link(self, link_path: Path, target: Path) -> Path:
        """Ensure ``link_path`` is a symlink to ``target``; record the dir."""
        link_path = link_path.expanduser()
        target = target.expanduser()
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.is_symlink():
            resolved = Path(os.readlink(link_path))
            if not resolved.is_absolute():
                resolved = (link_path.parent / resolved).resolve()
            if resolved != target.resolve():
                link_path.unlink()
            else:
                self._watch_dirs.add(link_path.parent)
                return link_path

        if link_path.exists():
            raise FileExistsError(
                f"Cannot place HaWoR shim symlink (real file exists): {link_path}"
            )

        link_path.symlink_to(target)
        self._created.append(link_path)
        self._watch_dirs.add(link_path.parent)
        return link_path

    def prune(self) -> None:
        """Delete any broken links we created, plus stale ones in tracked dirs."""
        for link_path in self._created:
            if link_path.is_symlink() and not link_path.exists():
                link_path.unlink(missing_ok=True)
        # ``_watch_dirs`` only ever includes dirs where we created a link, so
        # siblings there are removed only if they are also broken links.
        for watch_dir in self._watch_dirs:
            if not watch_dir.is_dir():
                continue
            for entry in watch_dir.iterdir():
                if entry.is_symlink() and not entry.exists():
                    try:
                        entry.unlink()
                    except OSError:
                        pass

    def reset(self) -> None:
        self._created.clear()
        self._watch_dirs.clear()


def _containing_dir(path: Path) -> Path:
    return path if path.is_dir() else path.parent


class symlink_tracker:
    """Context manager wrapping a :class:`SymlinkTracker` scoped to a block."""

    def __init__(self) -> None:
        self._tracker = SymlinkTracker()

    def __enter__(self) -> SymlinkTracker:
        return self._tracker

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tracker.prune()
