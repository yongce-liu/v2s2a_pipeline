#!/usr/bin/env bash
# Apply the main-repo-maintained patches to the official submodules.
#
# pkgs/* are official submodules; our local modifications live in
# patches/<name>.patch instead of being committed inside the submodule, so
# `git -C pkgs/<name> status` stays clean and upstream updates are a plain
# pointer bump (re-generate the patch if it no longer applies). For patches
# that generate derived artifacts after apply, fill in the REBUILD and
# EXCLUDES maps below keyed on the <name>.patch basename.
#
# Usage:
#   scripts/apply_patches.sh          # apply if needed (idempotent)
#   scripts/apply_patches.sh --check  # verify only; non-zero exit if unapplied
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHES_DIR="$ROOT/patches"

# Rebuild steps for patches that ship generation scripts the submodule runs
# after apply (symlink trees, .pth files). Associative array keyed on the
# patches/<name>.patch basename; the value runs as a command from the repo.
declare -A REBUILD=(
    [HaWoR]="python scripts/refresh_haworpkg.py"
)

check_or_apply() {
    local name="$1" repo="$2" patch="$3" check="$4"
    if [ ! -d "$ROOT/$repo" ]; then
        echo "[patches] SKIP $repo (submodule not checked out)" >&2
        return 0
    fi
    # Applying a main-repo patch necessarily dirties tracked files in the
    # submodule. Hide that worktree dirt from the superproject status while
    # still reporting submodule commit-pointer changes.
    git -C "$ROOT" config --local "submodule.$repo.ignore" dirty
    if git -C "$ROOT/$repo" apply --check --reverse "$patch" 2>/dev/null; then
        echo "[patches] ok: $repo already patched" >&2
    elif git -C "$ROOT/$repo" apply --check "$patch" 2>/dev/null; then
        if [ "$check" = "--check" ]; then
            echo "[patches] MISSING: $repo ($patch not applied)" >&2
            return 1
        fi
        git -C "$ROOT/$repo" apply "$patch"
        echo "[patches] applied $patch -> $repo"
    else
        echo "[patches] ERROR: $patch neither applies nor reverses cleanly in $repo" >&2
        return 1
    fi
    if [ "$check" != "--check" ]; then
        if [ -n "${REBUILD[$name]:-}" ]; then
            echo "[patches] rebuild $repo: ${REBUILD[$name]}"
            (cd "$ROOT/$repo" && eval "${REBUILD[$name]}")
        fi
    fi
}

rc=0
for patch_file in "$PATCHES_DIR"/*.patch; do
    [ -e "$patch_file" ] || { echo "[patches] no patches in $PATCHES_DIR" >&2; exit 0; }
    name="$(basename "$patch_file" .patch)"
    repo="pkgs/$name"
    check_or_apply "$name" "$repo" "$patch_file" "${1:-}" || rc=1
done
exit $rc
