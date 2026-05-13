from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import requests
from PIL import Image, ImageFilter, ImageOps

from services.config import load_env

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency at runtime
    fitz = None

load_env()

MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "5"))
MAX_IMAGE_SIDE = int(os.getenv("OCR_MAX_IMAGE_SIDE", "1600"))
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "0"))
DEFAULT_OCR_MODEL = os.getenv("OCR_MODEL", "gpt-4.1-mini")
DEFAULT_YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
DEFAULT_YANDEX_OCR_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"


def extract_text_from_file(path: Path, llm_mode: str = "local_llm") -> str:
    """Extract text from txt/image/pdf via an LLM OCR provider."""
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix not in {".jpg", ".jpeg", ".png", ".pdf"}:
        return f"Неподдерживаемый формат файла: {path.suffix}"

    settings = _ocr_api_settings(llm_mode)
    if not settings["enabled"]:
        return (
            "LLM OCR НЕ НАСТРОЕН.\n\n"
            "Сейчас проект ожидает AI-распознавание через vision-модель.\n"
            "Добавьте или проверьте настройки в .env:\n"
            "OCR_PROVIDER=openai_compatible\n"
            "OPENAI_BASE_URL=http://127.0.0.1:3000/v1\n"
            f"OCR_MODEL={DEFAULT_OCR_MODEL}\n\n"
            f"Файл: {path.name}"
        )

    try:
        return _extract_text_with_llm(path, settings)
    except Exception as exc:  # pragma: no cover
        return _ocr_error_message(path, str(exc))


def _extract_text_with_llm(path: Path, settings: dict[str, str | bool]) -> str:
    if path.suffix.lower() == ".pdf":
        images = _render_pdf_to_images(path)
    else:
        with Image.open(path) as img:
            images = [img.convert("RGB")]

    chunks: List[str] = []
    for page_index, image in enumerate(images, start=1):
        processed = _prepare_image(image)
        page_text = _run_vision_ocr(
            processed,
            page_number=page_index,
            total_pages=len(images),
            file_name=path.name,
            settings=settings,
        )
        chunks.append(f"--- Страница {page_index} ---\n{page_text.strip()}")

    return "\n\n".join(chunks).strip()


def _render_pdf_to_images(path: Path) -> List[Image.Image]:
    if fitz is None:
        raise RuntimeError("Для PDF нужен PyMuPDF. Установите зависимости из requirements.txt.")

    doc = fitz.open(path)
    images: List[Image.Image] = []
    try:
        for index in range(min(len(doc), MAX_PAGES)):
            page = doc.load_page(index)
            pix = page.get_pixmap(dpi=220, alpha=False)
            image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append(image)
    finally:
        doc.close()

    if not images:
        raise RuntimeError("PDF не содержит страниц для обработки.")

    return images


def _prepare_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.autocontrast(image)
    gray = ImageOps.grayscale(image)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))

    width, height = gray.size
    longest = max(width, height)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / float(longest)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        gray = gray.resize(new_size)

    return gray.convert("RGB")


def _run_vision_ocr(image: Image.Image, page_number: int, total_pages: int, file_name: str, settings: dict[str, str | bool]) -> str:
    if settings.get("mode") == "yandex_aistudio":
        return _run_yandex_vision_ocr(image, page_number, total_pages, file_name, settings)

    base_url = str(settings["base_url"]).rstrip("/")
    api_key = str(settings["api_key"])
    model = str(settings["model"])

    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL не задан.")

    image_b64 = _image_to_base64(image)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if settings.get("project"):
        headers["OpenAI-Project"] = str(settings["project"])
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 4000,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты ИИ для OCR школьных работ. Твоя задача — распознать рукописный или печатный текст с изображения. "
                    "Верни только текст страницы без пояснений. Сохраняй порядок чтения, переносы, абзацы, нумерацию, "
                    "математические символы и знаки препинания, если они различимы."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _build_ocr_prompt(
                            page_number=page_number,
                            total_pages=total_pages,
                            file_name=file_name,
                            image_size=image.size,
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ],
    }

    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=LLM_REQUEST_TIMEOUT or None,
    )
    response.raise_for_status()
    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Некорректный ответ OCR endpoint: {data}") from exc

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        content = "\n".join(part for part in parts if part)

    return str(content).strip()


def _run_yandex_vision_ocr(image: Image.Image, page_number: int, total_pages: int, file_name: str, settings: dict[str, str | bool]) -> str:
    endpoint = str(settings["base_url"]).strip()
    api_key = str(settings["api_key"])
    model = str(settings["model"])

    image_b64 = _image_to_base64(image)
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "mimeType": "PNG",
        "languageCodes": ["ru", "en"],
        "model": model,
        "content": image_b64,
    }

    response = requests.post(
        endpoint,
        headers=headers,
        data=json.dumps(payload),
        timeout=LLM_REQUEST_TIMEOUT or None,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"{exc}. Response body: {response.text[:2000]}") from exc

    data = response.json()
    text = _extract_yandex_ocr_text(data)
    if not text:
        raise RuntimeError(f"Yandex Vision OCR returned empty text. Response body: {json.dumps(data, ensure_ascii=False)[:2000]}")

    return text.strip()


def _extract_yandex_ocr_text(data: dict) -> str:
    annotation = data.get("result", {}).get("textAnnotation", {})
    full_text = annotation.get("fullText")
    if full_text:
        return str(full_text)

    lines: list[str] = []
    for block in annotation.get("blocks", []) or []:
        for line in block.get("lines", []) or []:
            line_text = line.get("text")
            if line_text:
                lines.append(str(line_text))
                continue
            words = [str(word.get("text", "")) for word in line.get("words", []) or [] if word.get("text")]
            if words:
                lines.append(" ".join(words))
    return "\n".join(lines)


def _build_ocr_prompt(page_number: int, total_pages: int, file_name: str, image_size: Tuple[int, int]) -> str:
    width, height = image_size
    return (
        "Распознай весь читаемый текст с этой страницы домашней работы. "
        "Не пересказывай и не объясняй содержимое. Если символ неуверенный, выбери наиболее вероятный вариант.\n\n"
        f"file_name: {file_name}\n"
        f"page_number: {page_number}\n"
        f"total_pages: {total_pages}\n"
        f"rendered_image_size: {width}x{height}\n"
        "document_type: school homework"
    )


def _image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _ocr_api_settings(llm_mode: str) -> dict[str, str | bool]:
    mode = (llm_mode or "local_llm").strip().lower()
    if mode == "yandex_aistudio":
        api_key = _real_api_key(os.getenv("YANDEX_API_KEY", ""))
        folder_id = os.getenv("YANDEX_FOLDER_ID", "").strip()
        base_url = os.getenv("YANDEX_OCR_BASE_URL", DEFAULT_YANDEX_OCR_URL).strip()
        model = os.getenv("YANDEX_OCR_MODEL", "handwritten").strip()
        return {
            "enabled": bool(api_key and folder_id and base_url and model),
            "mode": "yandex_aistudio",
            "base_url": base_url,
            "api_key": api_key,
            "project": folder_id,
            "model": model,
        }

    provider = os.getenv("OCR_PROVIDER", "").strip().lower()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    return {
        "enabled": provider in {"openai", "openai_compatible"} and bool(base_url),
        "mode": "local_llm",
        "base_url": base_url,
        "api_key": os.getenv("OPENAI_API_KEY", "EMPTY"),
        "project": "",
        "model": os.getenv("OCR_MODEL", DEFAULT_OCR_MODEL),
    }


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


def _ocr_error_message(path: Path, error: str) -> str:
    return (
        "ОШИБКА LLM OCR.\n\n"
        f"Файл: {path.name}\n"
        f"Причина: {error}\n\n"
        "Проверьте доступность vision-LLM endpoint и параметры в .env."
    )
