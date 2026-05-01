from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / 'data' / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)



def generate_json_report(submission: dict[str, Any]) -> Path:
    path = REPORTS_DIR / f"{submission['id']}.json"
    payload: dict[str, Any] = {
        'student_name': submission['student_name'],
        'work_type': submission['work_type'],
        'grade_level': submission['grade_level'],
        'criteria': submission['criteria'],
        'source_filename': submission['filename'],
        'text': submission['text'],
        'analysis': submission['analysis'],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path
