from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import uuid

from services.config import load_env
from services.ocr import extract_text_from_file
from services.analyzer import analyze_text
from services.storage import save_submission, list_submissions, get_submission
from services.report import generate_json_report

load_env()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.pdf', '.txt'}

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def current_ocr_settings() -> dict:
    provider = (os.getenv('OCR_PROVIDER', 'disabled') or 'disabled').strip()
    base_url = os.getenv('OPENAI_BASE_URL', '').strip()
    return {
        'provider': provider,
        'model': os.getenv('OCR_MODEL', 'allenai/olmOCR-7B-0225-preview'),
        'endpoint': base_url if base_url else 'not set',
        'enabled': provider in {'openai', 'openai_compatible'},
    }


@app.route('/')
def index():
    submissions = list_submissions()
    return render_template('index.html', submissions=submissions, ocr=current_ocr_settings())


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('work_file')
    student_name = request.form.get('student_name', 'Не указан').strip() or 'Не указан'
    work_type = request.form.get('work_type', 'Сочинение').strip() or 'Сочинение'
    grade_level = request.form.get('grade_level', '5-11').strip() or '5-11'
    criteria_raw = request.form.get('criteria', '').strip()

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
    extracted_text = extract_text_from_file(saved_path)
    analysis_result = analyze_text(
        extracted_text,
        student_name=student_name,
        work_type=work_type,
        grade_level=grade_level,
        criteria=criteria,
    )

    submission = {
        'id': file_id,
        'student_name': student_name,
        'work_type': work_type,
        'grade_level': grade_level,
        'criteria': criteria,
        'filename': file.filename,
        'stored_filename': saved_name,
        'text': extracted_text,
        'analysis': analysis_result,
        'ocr': current_ocr_settings(),
    }
    save_submission(submission)
    return redirect(url_for('result', submission_id=file_id))


@app.route('/result/<submission_id>')
def result(submission_id: str):
    submission = get_submission(submission_id)
    if not submission:
        flash('Работа не найдена.')
        return redirect(url_for('index'))
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
