import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs" / "base.yaml"


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path=None) -> dict:
    """Load base.yaml, then deep-merge the given override on top."""
    cfg = _load_yaml(BASE_CONFIG)
    if path is not None and Path(path).resolve() != BASE_CONFIG.resolve():
        cfg = _deep_merge(cfg, _load_yaml(path))
    return cfg


def flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested config into dot-keys for MLflow params."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten(v, prefix=f"{key}."))
        else:
            flat[key] = v
    return flat