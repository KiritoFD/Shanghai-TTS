"""Thin re-export shim — canonical implementation is in shared.config.

All callers should eventually migrate to ``from shared.config import load_runtime_config``.
This module exists for backward compatibility so that ``from path_config import ...`` keeps working.
"""
from __future__ import annotations

from shared.config import APP_ROOT, CONFIG_PATH, REPO_ROOT, load_runtime_config

# Keep the old name for any code that references path_config.ROOT
ROOT = APP_ROOT

__all__ = ["ROOT", "APP_ROOT", "REPO_ROOT", "CONFIG_PATH", "load_runtime_config"]
