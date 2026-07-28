"""
conftest.py — pytest root configuration for CalorieVision
──────────────────────────────────────────────────────────
Ensures the repo root is on sys.path so that `shared`, `cv_pipeline`,
`fusion_pipeline`, and `app` are importable from any test or script.

The previous hyphen-workaround (MetaPathFinder) has been removed.
Directories are now named with underscores (cv_pipeline, fusion_pipeline)
so Python's standard import machinery finds them without any tricks.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Always ensure repo root is on sys.path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
