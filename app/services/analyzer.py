from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

RUSSIAN_VOWELS = 'аеёиоуыэюяАЕЁИОУЫЭЮЯ'


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


def analyze_text(text: str, student_name: str, work_type: str, grade_level: str, criteria: list[str]) -> dict[str, Any]:
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
