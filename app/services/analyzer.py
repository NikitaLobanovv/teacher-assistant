from __future__ import annotations

import math
import json
import os
import re
import base64
from io import BytesIO
from pathlib import Path
from collections import Counter
from typing import Any

import requests
from PIL import Image, ImageOps

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency at runtime
    fitz = None

RUSSIAN_VOWELS = 'аеёиоуыэюяАЕЁИОУЫЭЮЯ'
DEFAULT_YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
DEFAULT_ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "yandexgpt/latest")
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "0"))
MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "5"))
MAX_IMAGE_SIDE = int(os.getenv("OCR_MAX_IMAGE_SIDE", "1600"))


def _is_service_message(text: str) -> bool:
    markers = ("LLM OCR НЕ НАСТРОЕН", "ОШИБКА LLM OCR", "Неподдерживаемый формат файла")
    return any(marker in text for marker in markers)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё'-]+", text)


def detect_repetitions(words: list[str]) -> list[dict[str, Any]]:
    normalized = [w.lower() for w in words if len(w) > 3]
    counts = Counter(normalized)
    return [
        {'word': word, 'count': count}
        for word, count in counts.most_common()
        if count >= 4
    ][:8]


def spelling_like_issues(text: str) -> list[dict[str, str]]:
    issues = []
    for match in re.finditer(r'\b[А-ЯЁ]{2,}[а-яё]+\b', text):
        issues.append({
            'type': 'орфография',
            'fragment': match.group(0),
            'hint': 'Возможно, слово начинается с лишних заглавных букв.'
        })
    for match in re.finditer(r'\b\w{20,}\b', text):
        issues.append({
            'type': 'стилистика',
            'fragment': match.group(0),
            'hint': 'Очень длинное слово — проверьте корректность распознавания или написания.'
        })
    return issues[:10]


def punctuation_like_issues(sentences: list[str]) -> list[dict[str, str]]:
    issues = []
    for sentence in sentences:
        if len(sentence.split()) > 18 and ',' not in sentence and ';' not in sentence:
            issues.append({
                'type': 'пунктуация',
                'fragment': sentence[:120],
                'hint': 'В длинном предложении могут отсутствовать знаки препинания.'
            })
        if sentence.count(',') >= 4:
            issues.append({
                'type': 'пунктуация',
                'fragment': sentence[:120],
                'hint': 'Проверьте перегруженность предложения запятыми.'
            })
    return issues[:10]


def estimate_readability(words: list[str], sentences: list[str]) -> float:
    if not words or not sentences:
        return 0.0
    syllables = 0
    for word in words:
        syllables += sum(1 for ch in word if ch in RUSSIAN_VOWELS)
    asl = len(words) / max(len(sentences), 1)
    asw = syllables / max(len(words), 1)
    score = 206.835 - 1.3 * asl - 60.1 * asw
    return round(max(0.0, min(100.0, score)), 1)


def score_text(words: list[str], sentences: list[str], issues_count: int, criteria_count: int) -> int:
    if not words:
        return 1
    base = 5.0
    if len(words) < 40:
        base -= 1.0
    if len(sentences) < 3:
        base -= 0.5
    base -= min(2.5, issues_count * 0.15)
    if criteria_count >= 3 and len(words) > 80:
        base += 0.5
    return max(1, min(5, int(round(base))))


def criterion_breakdown(criteria: list[str], words: list[str], sentences: list[str], issues_count: int) -> list[dict[str, Any]]:
    if not criteria:
        criteria = ['Грамотность', 'Логика изложения', 'Полнота ответа']

    items: list[dict[str, Any]] = []
    for idx, criterion in enumerate(criteria, start=1):
        score = 5
        explanation = 'Критерий выполнен на хорошем уровне.'

        lower = criterion.lower()
        if 'грам' in lower or 'орф' in lower or 'пункт' in lower:
            score = max(2, 5 - math.ceil(issues_count / 4))
            explanation = 'Оценка зависит от количества найденных орфографических и пунктуационных проблем.'
        elif 'лог' in lower or 'структ' in lower:
            score = 4 if len(sentences) >= 3 else 3
            explanation = 'Проверяется наличие вступления, основной части и завершения.'
        elif 'оригин' in lower or 'аргумент' in lower:
            score = 4 if len(words) > 70 else 3
            explanation = 'Оценивается глубина раскрытия темы и насыщенность текста аргументами.'
        elif 'полнот' in lower or 'соответ' in lower:
            score = 5 if len(words) > 80 else 4 if len(words) > 40 else 3
            explanation = 'Критерий оценивает, насколько полно раскрыта тема.'

        items.append({
            'id': idx,
            'criterion': criterion,
            'score': score,
            'explanation': explanation,
        })
    return items


def recommendations(words: list[str], repetitions: list[dict[str, Any]], issues_count: int) -> list[str]:
    tips = []
    if len(words) < 50:
        tips.append('Расширьте ответ: добавьте аргументы, примеры или пояснения.')
    if issues_count > 0:
        tips.append('Перепроверьте орфографию и пунктуацию в отмеченных фрагментах.')
    if repetitions:
        repeated = ', '.join(item['word'] for item in repetitions[:3])
        tips.append(f'Сократите повторы слов: {repeated}. Подберите синонимы.')
    if not tips:
        tips.append('Работа выглядит цельной. Можно усилить текст примерами и более точными формулировками.')
    return tips


def analyze_file_with_yandex(path: Path, student_name: str, work_type: str, grade_level: str, criteria: list[str]) -> tuple[str, dict[str, Any]]:
    settings = _analysis_api_settings("yandex_aistudio")
    if not settings["enabled"]:
        analysis = _analysis_error(
            "Yandex AI Studio is not configured. Add YANDEX_API_KEY and YANDEX_FOLDER_ID to .env.",
            criteria,
        )
        return "", analysis

    if path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text, analyze_text_with_llm(text, student_name, work_type, grade_level, criteria, "yandex_aistudio")

    try:
        images = _file_to_images(path)
        payload = {
            "model": _yandex_model("YANDEX_VISION_MODEL", "gemma-3-27b-it"),
            "temperature": 0.2,
            "max_tokens": 5000,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant for teachers checking student written work from images. "
                        "Read the attached pages and return only valid JSON with this shape: "
                        '{"recognized_text": str, "summary": str, '
                        '"detected_issues": [{"type": str, "fragment": str, "hint": str}], '
                        '"criteria_scores": [{"id": int, "criterion": str, "score": int, "explanation": str}], '
                        '"recommendations": [str], "feedback": str, '
                        '"metrics": {"words": int, "sentences": int, "readability": number, "repetitions": []}, "grade": int}. '
                        "Scores and grade must be integers from 1 to 5. Write all user-facing text in Russian."
                    ),
                },
                {
                    "role": "user",
                    "content": _build_file_analysis_content(images, path.name, student_name, work_type, grade_level, criteria),
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {settings['api_key']}",
            "OpenAI-Project": settings["project"],
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{settings['base_url'].rstrip('/')}/chat/completions",
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=LLM_REQUEST_TIMEOUT or None,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        result = _loads_json_content(content)
        recognized_text = str(result.pop("recognized_text", "")).strip()
        analysis = _normalize_llm_analysis(result, recognized_text, criteria)
        return recognized_text, analysis
    except Exception as exc:  # pragma: no cover
        analysis = _analysis_error(f"Yandex one-request analysis failed: {exc}", criteria)
        return "", analysis


def analyze_text_with_llm(text: str, student_name: str, work_type: str, grade_level: str, criteria: list[str], llm_mode: str) -> dict[str, Any]:
    if _is_service_message(text):
        return analyze_text_local(text, student_name, work_type, grade_level, criteria)

    settings = _analysis_api_settings(llm_mode)
    if not settings["enabled"]:
        return _analysis_error(
            "LLM analysis is not configured. Add YANDEX_API_KEY and YANDEX_FOLDER_ID for Yandex AI Studio or local OPENAI_BASE_URL/OCR_PROVIDER for local LLM.",
            criteria,
        )

    payload = {
        "model": settings["model"],
        "temperature": 0.2,
        "max_tokens": 3000,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant for teachers checking student written work. "
                    "Return only valid JSON with this shape: "
                    '{"summary": str, "detected_issues": [{"type": str, "fragment": str, "hint": str}], '
                    '"criteria_scores": [{"id": int, "criterion": str, "score": int, "explanation": str}], '
                    '"recommendations": [str], "feedback": str, '
                    '"metrics": {"words": int, "sentences": int, "readability": number, "repetitions": []}, "grade": int}. '
                    "Scores and grade must be integers from 1 to 5. Write all user-facing text in Russian."
                ),
            },
            {
                "role": "user",
                "content": _build_analysis_prompt(text, student_name, work_type, grade_level, criteria),
            },
        ],
    }
    if settings["mode"] == "yandex_aistudio":
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {settings['api_key']}",
        "Content-Type": "application/json",
    }
    if settings.get("project"):
        headers["OpenAI-Project"] = settings["project"]

    try:
        response = requests.post(
            f"{settings['base_url'].rstrip('/')}/chat/completions",
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=LLM_REQUEST_TIMEOUT or None,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        result = _loads_json_content(content)
        return _normalize_llm_analysis(result, text, criteria)
    except Exception as exc:  # pragma: no cover
        return _analysis_error(f"LLM analysis request failed: {exc}", criteria)


def _analysis_api_settings(llm_mode: str) -> dict[str, Any]:
    mode = (llm_mode or "local_llm").strip().lower()
    if mode == "yandex_aistudio":
        api_key = _real_api_key(os.getenv("YANDEX_API_KEY", ""))
        folder_id = os.getenv("YANDEX_FOLDER_ID", "").strip()
        base_url = os.getenv("YANDEX_BASE_URL", DEFAULT_YANDEX_BASE_URL).strip()
        model = _yandex_model("YANDEX_ANALYSIS_MODEL", "yandexgpt/latest")
        return {
            "enabled": bool(api_key and folder_id and base_url and model),
            "mode": "yandex_aistudio",
            "base_url": base_url,
            "api_key": api_key,
            "project": folder_id,
            "model": model,
        }

    provider = os.getenv("OCR_PROVIDER", "").strip().lower()
    base_url = os.getenv("ANALYSIS_LOCAL_BASE_URL", os.getenv("OPENAI_BASE_URL", "")).strip()
    return {
        "enabled": provider in {"openai", "openai_compatible"} and bool(base_url),
        "mode": "local_llm",
        "base_url": base_url,
        "api_key": os.getenv("ANALYSIS_LOCAL_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")),
        "project": "",
        "model": os.getenv("ANALYSIS_LOCAL_MODEL", os.getenv("OCR_MODEL", DEFAULT_ANALYSIS_MODEL)),
    }


def _build_analysis_prompt(text: str, student_name: str, work_type: str, grade_level: str, criteria: list[str]) -> str:
    criteria_text = "\n".join(f"{index}. {criterion}" for index, criterion in enumerate(criteria, start=1))
    if not criteria_text:
        criteria_text = "1. Грамотность\n2. Логика изложения\n3. Полнота ответа"

    return (
        f"Ученик: {student_name}\n"
        f"Тип работы: {work_type}\n"
        f"Класс: {grade_level}\n"
        f"Критерии проверки:\n{criteria_text}\n\n"
        "Проверь работу по критериям, найди орфографические, пунктуационные, логические и содержательные проблемы. "
        "Дай итоговую оценку от 1 до 5 и краткую обратную связь для ученика.\n\n"
        f"Текст работы:\n{text[:12000]}"
    )


def _loads_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "\n".join(
            item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )
    text = str(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM returned JSON that is not an object")
    return data


def _build_file_analysis_content(images: list[Image.Image], file_name: str, student_name: str, work_type: str, grade_level: str, criteria: list[str]) -> list[dict[str, Any]]:
    criteria_text = "\n".join(f"{index}. {criterion}" for index, criterion in enumerate(criteria, start=1))
    if not criteria_text:
        criteria_text = "1. Грамотность\n2. Логика изложения\n3. Полнота ответа"

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Файл: {file_name}\n"
                f"Ученик: {student_name}\n"
                f"Тип работы: {work_type}\n"
                f"Класс: {grade_level}\n"
                f"Критерии проверки:\n{criteria_text}\n\n"
                "Сначала распознай весь текст на всех приложенных страницах. Затем сразу проверь работу по критериям. "
                "Не делай отдельный OCR-ответ: верни один JSON с recognized_text и результатами проверки."
            ),
        }
    ]

    for index, image in enumerate(images, start=1):
        content.append({"type": "text", "text": f"Страница {index} из {len(images)}"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_image_to_base64(_prepare_image(image))}"},
        })
    return content


def _file_to_images(path: Path) -> list[Image.Image]:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png"}:
        with Image.open(path) as image:
            return [image.convert("RGB")]

    if suffix == ".pdf":
        if fitz is None:
            raise RuntimeError("Для PDF нужен PyMuPDF. Установите зависимости из requirements.txt.")
        doc = fitz.open(path)
        images: list[Image.Image] = []
        try:
            for index in range(min(len(doc), MAX_PAGES)):
                page = doc.load_page(index)
                pix = page.get_pixmap(dpi=220, alpha=False)
                images.append(Image.open(BytesIO(pix.tobytes("png"))).convert("RGB"))
        finally:
            doc.close()
        if images:
            return images

    raise RuntimeError(f"Неподдерживаемый формат файла: {path.suffix}")


def _prepare_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / float(longest)
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    return image


def _image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _normalize_llm_analysis(result: dict[str, Any], text: str, criteria: list[str]) -> dict[str, Any]:
    local_metrics = analyze_text_local(text, "", "", "", criteria)["metrics"]
    result["summary"] = str(result.get("summary", "Проверка выполнена через LLM."))
    result["detected_issues"] = _list_of_dicts(result.get("detected_issues"))
    result["criteria_scores"] = _normalize_criteria(result.get("criteria_scores"), criteria)
    result["recommendations"] = [str(item) for item in result.get("recommendations", []) if str(item).strip()][:8]
    if not result["recommendations"]:
        result["recommendations"] = ["Доработайте отмеченные места и проверьте финальную версию работы."]
    result["feedback"] = str(result.get("feedback", "Проверка выполнена автоматически."))
    result["metrics"] = result.get("metrics") if isinstance(result.get("metrics"), dict) else local_metrics
    result["metrics"]["words"] = int(result["metrics"].get("words", local_metrics["words"]))
    result["metrics"]["sentences"] = int(result["metrics"].get("sentences", local_metrics["sentences"]))
    result["metrics"]["readability"] = result["metrics"].get("readability", local_metrics["readability"])
    result["metrics"]["repetitions"] = result["metrics"].get("repetitions", local_metrics["repetitions"])
    result["grade"] = _clamp_score(result.get("grade", 1))
    return result


def _normalize_criteria(items: Any, criteria: list[str]) -> list[dict[str, Any]]:
    normalized = _list_of_dicts(items)
    if not normalized:
        normalized = criterion_breakdown(criteria, [], [], 0)

    for index, item in enumerate(normalized, start=1):
        item["id"] = int(item.get("id", index))
        item["criterion"] = str(item.get("criterion", criteria[index - 1] if index <= len(criteria) else f"Критерий {index}"))
        item["score"] = _clamp_score(item.get("score", 1))
        item["explanation"] = str(item.get("explanation", "Оценка выставлена автоматически."))
    return normalized


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 1
    return max(1, min(5, score))


def _real_api_key(value: str) -> str:
    value = (value or "").strip()
    return "" if value.upper() == "EMPTY" else value


def _yandex_model(env_name: str, default_slug: str) -> str:
    model = os.getenv(env_name, "").strip()
    if model:
        return model

    folder_id = os.getenv("YANDEX_FOLDER_ID", "").strip()
    if not folder_id:
        return ""
    return f"gpt://{folder_id}/{default_slug}"


def _analysis_error(error: str, criteria: list[str]) -> dict[str, Any]:
    return {
        "summary": "Не удалось выполнить проверку через LLM.",
        "detected_issues": [],
        "criteria_scores": criterion_breakdown(criteria, [], [], 0),
        "recommendations": [
            "Проверьте ключ API, адрес endpoint и название модели в .env.",
            "После настройки повторно загрузите работу и выберите нужный режим проверки.",
        ],
        "feedback": error,
        "metrics": {"words": 0, "sentences": 0, "readability": 0.0, "repetitions": []},
        "grade": 1,
    }


def analyze_text(text: str, student_name: str, work_type: str, grade_level: str, criteria: list[str], llm_mode: str = "local_llm") -> dict[str, Any]:
    mode = (llm_mode or "local_llm").strip().lower()
    if mode in {"local_llm", "yandex_aistudio"}:
        return analyze_text_with_llm(
            text,
            student_name=student_name,
            work_type=work_type,
            grade_level=grade_level,
            criteria=criteria,
            llm_mode=mode,
        )
    return analyze_text_local(text, student_name, work_type, grade_level, criteria)


def analyze_text_local(text: str, student_name: str, work_type: str, grade_level: str, criteria: list[str]) -> dict[str, Any]:
    if _is_service_message(text):
        return {
            'summary': 'Не удалось получить распознанный текст от OCR-модуля.',
            'detected_issues': [],
            'criteria_scores': criterion_breakdown(criteria, [], [], 0),
            'recommendations': [
                'Проверьте настройки .env и доступность vision-модели.',
                'Повторно загрузите изображение после настройки OCR endpoint.',
            ],
            'feedback': 'Система не смогла выполнить AI-распознавание текста, поэтому полноценная проверка работы недоступна.',
            'metrics': {
                'words': 0,
                'sentences': 0,
                'readability': 0.0,
                'repetitions': [],
            },
            'grade': 1,
        }

    sentences = split_sentences(text)
    words = split_words(text)
    repetitions = detect_repetitions(words)
    issues = spelling_like_issues(text) + punctuation_like_issues(sentences)
    readability = estimate_readability(words, sentences)
    grade = score_text(words, sentences, len(issues), len(criteria))
    criteria_scores = criterion_breakdown(criteria, words, sentences, len(issues))
    recs = recommendations(words, repetitions, len(issues))

    summary = (
        f'Проверена работа ученика {student_name}. '
        f'Тип работы: {work_type}. '
        f'Обнаружено проблемных мест: {len(issues)}. '
        f'Предварительная итоговая оценка: {grade}/5.'
    )

    feedback = (
        f'Работа по типу "{work_type}" для уровня {grade_level} класса проверена автоматически. '
        f'Сильные стороны: {"достаточный объём текста" if len(words) > 70 else "понятная основная мысль"}. '
        f'Зоны роста: {"пунктуация и структура предложений" if len(issues) else "углубление аргументации"}. '
        'Итоговая рекомендация: доработать отмеченные места и затем повторно отправить работу на проверку.'
    )

    return {
        'summary': summary,
        'detected_issues': issues,
        'criteria_scores': criteria_scores,
        'recommendations': recs,
        'feedback': feedback,
        'metrics': {
            'words': len(words),
            'sentences': len(sentences),
            'readability': readability,
            'repetitions': repetitions,
        },
        'grade': grade,
    }
