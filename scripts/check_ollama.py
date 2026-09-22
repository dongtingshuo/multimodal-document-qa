"""Run one local multimodal smoke check, without any cloud requests."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw, ImageFont

from docqa.config import Settings
from docqa.generation import (
    SYSTEM_PROMPT,
    OllamaGenerator,
    evidence_context,
    generation_status,
    validate_citations,
)
from docqa.schemas import Element, Evidence, Source
from docqa.store import Store


def main():
    directory = ROOT / 'data' / 'ollama-check'
    directory.mkdir(parents=True, exist_ok=True)
    settings = Settings(data_dir=directory, generation_provider='ollama', api_key='', api_timeout=180)
    status = generation_status(settings)
    if status['status'] != 'available':
        raise SystemExit(status['message'])
    image = Image.new('RGB', (650, 320), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 36)
    for y, text in [(35, 'Quarter      Sales'), (120, 'Q1                12'), (205, 'Q2                18')]:
        draw.text((45, y), text, fill='black', font=font)
    image_path = directory / 'table.png'
    image.save(image_path)
    evidence = Evidence(
        element=Element(id='local-smoke', document_id='synthetic', kind='table', text='Sales table',
                        image_path=str(image_path), sources=[Source(page=1, bbox=(0, 0, 1, 1))]),
        document_name='合成表格', page_label='图片页码', matched_text='Sales table. Read the attached image.',
        rank=1, channels=['test_fixture'], scores={},
    )
    content, context, image_count = evidence_context([evidence])
    content.append({'type': 'text', 'text': '图片表格中 Q2 的 Sales 数值是多少？请用中文简短回答并引用 [E1]。'})
    started = time.perf_counter()
    result = OllamaGenerator(settings, Store(directory)).complete([
        {'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content},
    ])
    answer = result['choices'][0]['message']['content']
    refs, invalid, missing = validate_citations(answer, {'E1'})
    report = {
        'model': settings.ollama_model, 'provider': 'ollama', 'seconds': time.perf_counter() - started,
        'image_count': image_count, 'context': context, 'answer': answer,
        'usage': result.get('usage'), 'ollama': result.get('ollama'),
        'expected_value': '18', 'expected_value_present': '18' in answer,
        'citations_valid': not invalid and not missing and bool(refs),
        'cloud_calls': 0, 'scope': '合成图片单题连通性检查，不作为正式答案准确率',
    }
    (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report['expected_value_present'] and report['citations_valid'] else 1


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    raise SystemExit(main())
