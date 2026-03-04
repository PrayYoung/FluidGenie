from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

T = TypeVar("T")


def save_config_json(out_dir: str | Path, payload: Dict[str, Any], filename: str = "config.json") -> Path:
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    cfg_path = out_path / filename
    cfg_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return cfg_path


def load_config_json(ckpt_path: str, filename: str = "config.json") -> Optional[Dict[str, Any]]:
    if not ckpt_path:
        return None
    p = Path(ckpt_path).expanduser().resolve()
    candidates = []
    if p.is_dir():
        candidates.append(p / filename)
        candidates.append(p / "meta.json")
        candidates.append(p.parent / filename)
        candidates.append(p.parent / "meta.json")
    else:
        candidates.append(p.parent / filename)
        candidates.append(p.parent / "meta.json")
    for c in candidates:
        if c.exists():
            try:
                return json.loads(c.read_text())
            except Exception:
                return None
    return None


def merge_dataclass_from_config(obj: T, cfg: Dict[str, Any], defaults: T) -> T:
    """
    Merge config into obj by replacing only fields that are still at default values.
    """
    data = asdict(obj)
    for k, v in cfg.items():
        if k not in data:
            continue
        if getattr(obj, k) == getattr(defaults, k):
            data[k] = v
    return replace(obj, **data)
