from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # Python 3.10 fallback


def load_toml_config(path: str, section: str) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    cfg = data.get(section, {})
    if not isinstance(cfg, dict):
        raise ValueError(f"Config section [{section}] must be a table.")
    return cfg


def apply_config_defaults(args, defaults: Dict[str, Any], cfg: Dict[str, Any]):
    """
    If arg value equals its default and config provides a value, override it.
    """
    for k, default in defaults.items():
        if getattr(args, k) == default and k in cfg:
            setattr(args, k, cfg[k])
