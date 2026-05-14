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
LLM_REQUEST_TIMEOUT=0
HISTORY_LIMIT=20

ANALYSIS_LOCAL_BASE_URL=http://127.0.0.1:3000/v1
ANALYSIS_LOCAL_API_KEY=EMPTY
ANALYSIS_LOCAL_MODEL=local-model

YANDEX_BASE_URL=https://ai.api.cloud.yandex.net/v1
YANDEX_API_KEY=
YANDEX_FOLDER_ID=
YANDEX_OCR_BASE_URL=https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText
YANDEX_OCR_MODEL=handwritten
YANDEX_ANALYSIS_MODEL=
```

В режиме Yandex приложение делает два запроса: сначала Yandex Vision OCR распознаёт рукописный текст моделью `handwritten`, затем Yandex AI Studio анализирует распознанный текст и возвращает оценку, проблемные места и рекомендации.

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
