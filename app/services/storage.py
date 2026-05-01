from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / 'data' / 'submissions.json'
DB_PATH.parent.mkdir(parents=True, exist_ok=True)



def _load() -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    return json.loads(DB_PATH.read_text(encoding='utf-8'))



def _save(items: list[dict[str, Any]]) -> None:
    DB_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')



def save_submission(submission: dict[str, Any]) -> None:
    items = _load()
    items.insert(0, submission)
    _save(items)



def list_submissions() -> list[dict[str, Any]]:
    return _load()



def get_submission(submission_id: str) -> dict[str, Any] | None:
    for item in _load():
        if item['id'] == submission_id:
            return item
    return None
