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
DEFAULT_OCR_MODEL = os.getenv("OCR_MODEL", "gpt-4.1-mini")


def extract_text_from_file(path: Path) -> str:
    """Extract text from txt/image/pdf via an LLM OCR provider."""
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix not in {".jpg", ".jpeg", ".png", ".pdf"}:
        return f"Неподдерживаемый формат файла: {path.suffix}"

    provider = os.getenv("OCR_PROVIDER", "").strip().lower()
    if provider not in {"openai", "openai_compatible"}:
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
        return _extract_text_with_llm(path)
    except Exception as exc:  # pragma: no cover
        return _ocr_error_message(path, str(exc))


def _extract_text_with_llm(path: Path) -> str:
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


def _run_vision_ocr(image: Image.Image, page_number: int, total_pages: int, file_name: str) -> str:
    base_url = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    model = os.getenv("OCR_MODEL", DEFAULT_OCR_MODEL)

    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL не задан.")

    image_b64 = _image_to_base64(image)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
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
        timeout=180,
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


def _ocr_error_message(path: Path, error: str) -> str:
    return (
        "ОШИБКА LLM OCR.\n\n"
        f"Файл: {path.name}\n"
        f"Причина: {error}\n\n"
        "Проверьте доступность vision-LLM endpoint и параметры в .env."
    )
