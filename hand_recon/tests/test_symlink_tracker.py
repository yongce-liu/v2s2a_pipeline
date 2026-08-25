"""Tests for the symlink cleanup helpers."""

from __future__ import annotations

from pathlib import Path

from hand_recon.symlink_tracker import SymlinkTracker, symlink_tracker


def test_link_creates_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()

    with symlink_tracker() as tracker:
        link = tracker.link(tmp_path / "link", target)
        assert link.is_symlink()
        assert link.resolve() == target.resolve()

    # After the tracker exits the watch-dir prune runs; the live symlink
    # should survive because its target still exists.
    assert link.is_symlink()


def test_link_replaces_stale_symlink(tmp_path: Path) -> None:
    old_target = tmp_path / "old"
    old_target.mkdir()
    new_target = tmp_path / "new"
    new_target.mkdir()

    link_path = tmp_path / "link"
    link_path.symlink_to(old_target)

    tracker = SymlinkTracker()
    tracker.link(link_path, new_target)
    assert link_path.resolve() == new_target.resolve()


def test_link_refuses_to_replace_real_file(tmp_path: Path) -> None:
    (tmp_path / "real").write_text("content")
    tracker = SymlinkTracker()
    try:
        tracker.link(tmp_path / "real", tmp_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected FileExistsError")


def test_prune_removes_broken_links_only(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()

    working = tmp_path / "working"
    working.symlink_to(target)

    broken_dir = tmp_path / "broken_dir"
    broken_dir.symlink_to(tmp_path / "missing")

    tracker = SymlinkTracker()
    tracker.link(tmp_path / "tracked_order", target)  # records tmp_path
    tracker.prune()

    assert working.is_symlink()
    # broken_dir sits in a watched dir and is broken, so prune removes it.
    assert not broken_dir.is_symlink()
    assert not broken_dir.exists()


def test_prune_removes_broken_links_in_watched_dir(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()

    working = tmp_path / "sub" / "working"
    working.parent.mkdir(parents=True)
    working.symlink_to(target)

    broken_sibling = tmp_path / "sub" / "broken_sibling"
    broken_sibling.symlink_to(tmp_path / "missing")

    tracker = SymlinkTracker()
    tracker.link(working, target)
    tracker.prune()

    assert working.is_symlink()
    assert not broken_sibling.exists() and not broken_sibling.is_symlink()
