"""
conftest.py — pytest root configuration for CalorieVision
──────────────────────────────────────────────────────────
Adds both the repo root and the cv-pipeline directory to sys.path so that
imports like:

    from cv_pipeline.keypoints.extract_keypoints import extract_keypoints

work regardless of how pytest is invoked (from root or from a subdirectory).

The cv-pipeline directory is mapped to the importable name "cv_pipeline"
by inserting it with that alias in sys.path manipulation below.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

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

        if candidate_dir.is_dir():
            init = candidate_dir / "__init__.py"
            origin = str(init) if init.exists() else None
            spec = importlib.machinery.ModuleSpec(
                fullname,
                None,
                origin=origin,
                is_package=True,
            )
            spec.submodule_search_locations = [str(candidate_dir)]
            return spec

        if candidate_file and candidate_file.exists():
            loader = importlib.machinery.SourceFileLoader(
                fullname, str(candidate_file)
            )
            return importlib.util.spec_from_file_location(
                fullname, str(candidate_file), loader=loader
            )

        return None


# Register the finder once
if not any(isinstance(f, _HyphenFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _HyphenFinder())
