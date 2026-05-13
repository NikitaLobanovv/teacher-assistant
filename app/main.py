from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory, flash
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import uuid

from services.config import load_env
from services.ocr import extract_text_from_file
from services.analyzer import analyze_text, analyze_file_with_yandex
from services.storage import save_submission, list_submissions, get_submission
from services.report import generate_json_report

load_env()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.pdf', '.txt'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def current_ocr_settings(llm_mode: str = 'local_llm') -> dict:
    if llm_mode == 'yandex_aistudio':
        base_url = os.getenv('YANDEX_BASE_URL', 'https://ai.api.cloud.yandex.net/v1').strip()
        api_key = os.getenv('YANDEX_API_KEY', '').strip()
        folder_id = os.getenv('YANDEX_FOLDER_ID', '').strip()
        return {
            'provider': 'yandex_aistudio',
            'model': os.getenv('YANDEX_VISION_MODEL') or (f'gpt://{folder_id}/gemma-3-27b-it' if folder_id else 'not set'),
            'endpoint': base_url if base_url else 'not set',
            'enabled': bool(api_key and api_key.upper() != 'EMPTY' and folder_id),
        }

    provider = (os.getenv('OCR_PROVIDER', 'disabled') or 'disabled').strip()
    base_url = os.getenv('OPENAI_BASE_URL', '').strip()
    return {
        'provider': provider,
        'model': os.getenv('OCR_MODEL', 'allenai/olmOCR-7B-0225-preview'),
        'endpoint': base_url if base_url else 'not set',
        'enabled': provider in {'openai', 'openai_compatible'},
    }


def current_llm_settings() -> dict:
    local_base_url = os.getenv('ANALYSIS_LOCAL_BASE_URL', os.getenv('OPENAI_BASE_URL', '')).strip()
    yandex_base_url = os.getenv('YANDEX_BASE_URL', 'https://ai.api.cloud.yandex.net/v1').strip()
    yandex_key = os.getenv('YANDEX_API_KEY', '').strip()
    yandex_folder_id = os.getenv('YANDEX_FOLDER_ID', '').strip()
    return {
        'local_model': os.getenv('ANALYSIS_LOCAL_MODEL', os.getenv('OCR_MODEL', 'local-model')),
        'local_endpoint': local_base_url if local_base_url else 'not set',
        'yandex_model': os.getenv('YANDEX_ANALYSIS_MODEL') or (f'gpt://{yandex_folder_id}/yandexgpt/latest' if yandex_folder_id else 'not set'),
        'yandex_endpoint': yandex_base_url if yandex_base_url else 'not set',
        'yandex_enabled': bool(yandex_key and yandex_key.upper() != 'EMPTY' and yandex_folder_id),
    }


@app.route('/')
def index():
    submissions = list_submissions()
    return render_template('index.html', submissions=submissions, ocr=current_ocr_settings('yandex_aistudio'), llm=current_llm_settings())


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('work_file')
    student_name = request.form.get('student_name', 'Не указан').strip() or 'Не указан'
    work_type = request.form.get('work_type', 'Письменная работа').strip() or 'Письменная работа'
    grade_level = request.form.get('grade_level', '5-11').strip() or '5-11'
    criteria_raw = request.form.get('criteria', '').strip()
    llm_mode = request.form.get('llm_mode', 'yandex_aistudio').strip()
    if llm_mode not in {'local_llm', 'yandex_aistudio'}:
        llm_mode = 'yandex_aistudio'

    if not file or not file.filename:
        flash('Выберите файл для загрузки.')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Поддерживаются только .jpg, .jpeg, .png, .pdf и .txt')
        return redirect(url_for('index'))

    ext = Path(file.filename).suffix.lower()
    file_id = str(uuid.uuid4())
    saved_name = f'{file_id}{ext}'
    saved_path = UPLOAD_DIR / secure_filename(saved_name)
    file.save(saved_path)

    criteria = [item.strip() for item in criteria_raw.splitlines() if item.strip()]
    if llm_mode == 'yandex_aistudio':
        extracted_text, analysis_result = analyze_file_with_yandex(
            saved_path,
            student_name=student_name,
            work_type=work_type,
            grade_level=grade_level,
            criteria=criteria,
        )
    else:
        extracted_text = extract_text_from_file(saved_path, llm_mode=llm_mode)
        analysis_result = analyze_text(
            extracted_text,
            student_name=student_name,
            work_type=work_type,
            grade_level=grade_level,
            criteria=criteria,
            llm_mode=llm_mode,
        )

    submission = {
        'id': file_id,
        'student_name': student_name,
        'work_type': work_type,
        'grade_level': grade_level,
        'criteria': criteria,
        'filename': file.filename,
        'stored_filename': saved_name,
        'is_image': is_image_file(saved_name),
        'text': extracted_text,
        'analysis': analysis_result,
        'llm_mode': llm_mode,
        'ocr': current_ocr_settings(llm_mode),
    }
    save_submission(submission)
    return redirect(url_for('result', submission_id=file_id))


@app.route('/uploads/<path:filename>')
def uploaded_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route('/result/<submission_id>')
def result(submission_id: str):
    submission = get_submission(submission_id)
    if not submission:
        flash('Работа не найдена.')
        return redirect(url_for('index'))
    submission['is_image'] = submission.get('is_image', is_image_file(submission.get('stored_filename', '')))
    return render_template('result.html', submission=submission)


@app.route('/report/<submission_id>.json')
def report(submission_id: str):
    submission = get_submission(submission_id)
    if not submission:
        flash('Работа не найдена.')
        return redirect(url_for('index'))
    path = generate_json_report(submission)
    return send_file(path, as_attachment=True, download_name=f'report_{submission_id}.json')


if __name__ == '__main__':
    app.run(debug=True)
