- веб-интерфейс на Flask;
- загрузка работ (`.jpg`, `.jpeg`, `.png`, `.pdf`, `.txt`);
- OCR через vision-LLM;
- рендер PDF-страниц в изображения;
- модуль текстового анализа;
- предварительная оценка по критериям;
- обратная связь и рекомендации;
- сохранение истории проверок;
- экспорт JSON-отчёта.

## Быстрый старт

```bash
cd app
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r ../requirements.txt
cp ../.env.example ../.env
python main.py
```

После запуска `http://127.0.0.1:5000`.

### Переменные окружения

```bash
OCR_PROVIDER=openai_compatible
OPENAI_BASE_URL=http://127.0.0.1:3000/v1
OPENAI_API_KEY=EMPTY
OCR_MODEL=allenai/olmOCR-7B-0225-preview
OCR_MAX_PAGES=5
OCR_MAX_IMAGE_SIDE=1024
```

## Где находится интеграция

Файл: `app/services/ocr.py`

Ключевые функции:

- `extract_text_from_file()` — точка входа;
- `_render_pdf_to_images()` — рендер PDF;
- `_run_vision_ocr()` — отправка изображения в vision-LLM;
- `_build_olmocr_prompt()` — OCR prompt с метаданными страницы.

## Структура

```text
teacher_ai_assistant/
├── .env.example
├── app/
│   ├── main.py
│   ├── services/
│   ├── static/
│   ├── templates/
│   ├── data/
│   └── uploads/
├── requirements.txt
└── README.md
```
