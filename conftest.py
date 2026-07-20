"""
conftest.py — pytest root configuration for CalorieVision
──────────────────────────────────────────────────────────
<<<<<<< HEAD
Adds both the repo root and the cv-pipeline directory to sys.path so that
imports like:

    from cv_pipeline.keypoints.extract_keypoints import extract_keypoints

work regardless of how pytest is invoked (from root or from a subdirectory).

The cv-pipeline directory is mapped to the importable name "cv_pipeline"
by inserting it with that alias in sys.path manipulation below.
"""

=======
Registers MetaPathFinders so that hyphenated directory names work as
importable Python packages:

    cv-pipeline/   →  importable as  cv_pipeline.*
    fusion-pipeline/ → importable as  fusion_pipeline.*

This lets tests use clean dotted imports without any manual sys.path hacks.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
>>>>>>> b/youtube-download
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

<<<<<<< HEAD
# Add repo root so `shared`, `app`, etc. are importable
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add cv-pipeline/ as a path entry so its sub-packages (keypoints, eval, …)
# are importable.  Because the directory is named with a hyphen, we create a
# thin alias: we add a "cv_pipeline" namespace pointing at "cv-pipeline/".
import importlib
import importlib.abc
import importlib.machinery
import types


class _HyphenFinder(importlib.abc.MetaPathFinder):
    """Makes `import cv_pipeline.x` resolve to <repo>/cv-pipeline/x."""

    _base = REPO_ROOT / "cv-pipeline"

    def find_spec(self, fullname, path, target=None):
        parts = fullname.split(".")
        if parts[0] != "cv_pipeline":
            return None

        # Build the filesystem path for the rest of the dotted name
        rel = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        candidate_dir  = self._base / rel
        candidate_file = self._base / Path(*parts[1:-1]) / (parts[-1] + ".py") \
                         if len(parts) > 1 else None
=======
# Always ensure repo root is on sys.path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _HyphenatedDirFinder(importlib.abc.MetaPathFinder):
    """
    Maps a Python package name (underscores) to a filesystem directory
    (hyphens) that cannot itself be a package due to the hyphen.

    e.g.  import cv_pipeline.keypoints.foo
          resolves to  <repo>/cv-pipeline/keypoints/foo.py
    """

    def __init__(self, import_name: str, fs_path: Path) -> None:
        self._prefix = import_name          # e.g. "cv_pipeline"
        self._base   = fs_path              # e.g. Path(".../cv-pipeline")

    def find_spec(self, fullname, path, target=None):
        parts = fullname.split(".")
        if parts[0] != self._prefix:
            return None

        rel_parts = parts[1:]  # everything after the top-level alias

        # ── Package (directory with optional __init__.py) ──────────────────
        if rel_parts:
            candidate_dir = self._base.joinpath(*rel_parts)
        else:
            candidate_dir = self._base
>>>>>>> b/youtube-download

        if candidate_dir.is_dir():
            init = candidate_dir / "__init__.py"
            origin = str(init) if init.exists() else None
            spec = importlib.machinery.ModuleSpec(
                fullname,
<<<<<<< HEAD
                None,
=======
                loader=None,
>>>>>>> b/youtube-download
                origin=origin,
                is_package=True,
            )
            spec.submodule_search_locations = [str(candidate_dir)]
            return spec

<<<<<<< HEAD
        if candidate_file and candidate_file.exists():
            loader = importlib.machinery.SourceFileLoader(
                fullname, str(candidate_file)
            )
            return importlib.util.spec_from_file_location(
                fullname, str(candidate_file), loader=loader
            )
=======
        # ── Module (plain .py file) ────────────────────────────────────────
        if rel_parts:
            candidate_file = self._base.joinpath(*rel_parts[:-1], rel_parts[-1] + ".py")
            if candidate_file.exists():
                loader = importlib.machinery.SourceFileLoader(
                    fullname, str(candidate_file)
                )
                return importlib.util.spec_from_file_location(
                    fullname, str(candidate_file), loader=loader
                )
>>>>>>> b/youtube-download

        return None


<<<<<<< HEAD
# Register the finder once
if not any(isinstance(f, _HyphenFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _HyphenFinder())
=======
# Register finders for both hyphenated pipeline directories
_FINDERS = [
    _HyphenatedDirFinder("cv_pipeline",     REPO_ROOT / "cv-pipeline"),
    _HyphenatedDirFinder("fusion_pipeline", REPO_ROOT / "fusion-pipeline"),
]

for _finder in _FINDERS:
    if not any(
        isinstance(f, _HyphenatedDirFinder) and f._prefix == _finder._prefix
        for f in sys.meta_path
    ):
        sys.meta_path.insert(0, _finder)
>>>>>>> b/youtube-download
