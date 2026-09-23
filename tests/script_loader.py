"""Helper for importing the numbered scripts (which are not valid Python module
names) directly from the `scripts/` directory in tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def load_script(filename: str) -> ModuleType:
    """Import a script such as '02_logistic_regression.py' as a module."""
    path = SCRIPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")

    module_name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
