from __future__ import annotations

import os
from pathlib import Path


def load_env() -> None:
    """Simple .env loader without external dependencies."""
    current = Path(__file__).resolve()
    candidates = [
        current.parents[2] / '.env',
        current.parents[1] / '.env',
        Path.cwd() / '.env',
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding='utf-8', errors='ignore').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
        break
