"""Bounded source enrichment and conservative checks, not a semantic truth oracle."""
import hashlib
import os
import re
import tempfile
from decimal import Decimal
from pathlib import Path

import pymupdf as fitz

from .numeric_evidence import calculation_issues, normalize_math_text, percentage_issues, table_claim_issues
from .page_links import table_neighbors


def page_continuations(anchor, elements):
    if anchor.kind == 'table':
        return table_neighbors(anchor, elements)
    if anchor.kind != 'text' or len(anchor.sources) != 1:
        return []
    source = anchor.sources[0]
    text = anchor.text.strip()
    if source.bbox[3] < .75 or not text or re.search(r'[。！？.!?；;]$', text):
        return []
    candidates = sorted(
        (e for e in elements if e.document_id == anchor.document_id and e.kind == 'text'
         and len(e.sources) == 1 and e.sources[0].page == source.page + 1
         and .03 < e.sources[0].bbox[1] < .25 and len(e.text.strip()) > 3),
        key=lambda e: (e.sources[0].bbox[1], e.sources[0].bbox[0]),
    )
    for candidate in candidates:
        head = candidate.text.strip()
        # Skip page headers, but never jump across a new section heading.
        if re.match(r'^(?:第.+页|\d+)$', head):
            continue
        if re.match(r'^(?:第[一二三四五六七八九十\d]+[章节]|[一二三四五六七八九十]+、|[（(][一二三四五六七八九十]+[）)])', head):
            return []
        if anchor.title_path and candidate.title_path and anchor.title_path != candidate.title_path:
            return []
        a, _, c, _ = source.bbox
        x, _, z, _ = candidate.sources[0].bbox
        if min(c, z) - max(a, x) < .5 * min(c-a, z-x):
            continue
        # Require syntactic continuation: unfinished clause followed by a value,
        # or an explicitly open separator. Do not append arbitrary next pages.
        if re.match(r'^\d+(?:\.\d+)?(?:[:：%％‰]|\s*[万亿年月日人元])', head) or text.endswith(('，', ',', '：', ':')) or (anchor.title_path and anchor.title_path == candidate.title_path and not re.search(r'[。！？.!?；;]$', text)):
            return [candidate]
        return []
    return []


def source_image(item, store, cache_dir):
    """Render a bounded high-resolution crop from source; retain legacy fallback."""
    original = item.element.image_path
    if not original:
        return None
    doc = store.document(item.element.document_id) if store is not None else None
    if not doc:
        return original
    source = Path(doc.get('pdf_path') or doc.get('original_path') or '')
    if source.suffix.lower() != '.pdf' or not source.is_file():
        return original
    position = item.element.sources[0]
    stamp = source.stat()
    key = hashlib.sha256(repr((str(source.resolve()), stamp.st_mtime_ns, stamp.st_size,
                              position.page, position.bbox, 'source-crop-v1')).encode()).hexdigest()
    cache_dir = Path(cache_dir) / item.element.document_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / (key + '.png')
    if target.exists():
        return str(target)
    try:
        with fitz.open(source) as pdf:
            page = pdf[position.page - 1]
            x0, y0, x1, y1 = position.bbox
            clip = fitz.Rect(x0*page.rect.width, y0*page.rect.height,
                             x1*page.rect.width, y1*page.rect.height)
            if clip.is_empty:
                return original
            scale = min(4, 1600 / max(clip.width, clip.height))
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
            # Unique temporary paths allow simultaneous requests for one crop.
            handle, temp_name = tempfile.mkstemp(dir=cache_dir, suffix='.tmp')
            try:
                with os.fdopen(handle, 'wb') as output:
                    output.write(pix.tobytes('png'))
                Path(temp_name).replace(target)
            finally:
                Path(temp_name).unlink(missing_ok=True)
        return str(target)
    except (OSError, ValueError, IndexError, RuntimeError):
        return original


def preferred_image_ids(evidence, question, budget):
    figure = re.search(r'图\s*(\d+)', question or '')
    def priority(item):
        heading = re.search(r'图\s*(\d+)', item.element.text)
        exact = figure and heading and figure[1] == heading[1]
        return (not bool(exact), item.element.kind not in {'image', 'table'}, item.rank)
    visual = [e for e in evidence if e.element.image_path]
    return {e.element.id for e in sorted(visual, key=priority)[:budget]}


def numeric_values(text):
    return {Decimal(n.replace(',', '')) for n in re.findall(
        r'(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?', text)}


def _fragment_issues(text, context, evidence):
    """Detect clear textual mismatches; image contents require visual review."""
    refs = set(re.findall(r'\[(E\d+)\]', text))
    selected = [c for c in context if c['label'] in refs]
    if not selected:
        return []
    issues = []
    cleaned = re.sub(r'\[E\d+\]', '', text)
    claimed = numeric_values(cleaned)
    raw = {f'E{i}': e.element.text.strip() for i, e in enumerate(evidence, 1)}
    if claimed and all(re.match(r'^图\s*\d+', raw[c['label']])
                       and len(raw[c['label']]) < 100 and not c['image_sent'] for c in selected):
        figure_ids = {re.match(r'^图\s*(\d+)', raw[c['label']])[1] for c in selected}
        candidates = [c['label'] for c in context if c['image_sent']
                      and (m := re.match(r'^图\s*(\d+)', raw[c['label']])) and m[1] in figure_ids]
        hint = '当前已发送的对应图像编号为' + '、'.join(f'[{c}]' for c in candidates) if candidates else ''
        issues.append('引用只指向图题，不能支持图中具体数值；请引用实际图像证据。' + hint)
    if claimed and not any(c['image_sent'] for c in selected):
        available = numeric_values('\n'.join(c['text'] for c in selected))
        # Permit elementary differences/sums when both operands occur in cited
        # text. This is only a numeric consistency check, not unit/scope proof.
        arithmetic = {a-b for a in available for b in available} if len(available) <= 80 else set()
        arithmetic |= {a+b for a in available for b in available} if len(available) <= 80 else set()
        missing = claimed - available - arithmetic
        # Figure/table indices and date attribution are handled by provenance;
        # focus this conservative check on fractional values absent from text.
        missing = {v for v in missing if v != v.to_integral_value()}
        if missing:
            issues.append('引用文字中找不到这些回答数值：' + '、'.join(str(v) for v in sorted(missing)))
            for value in sorted(missing):
                locations = [f"[{c['label']}]" for c in context
                             if value in numeric_values(c['text']) and c['label'] not in refs]
                if locations:
                    issues.append(f'{value}出现在' + '、'.join(locations)
                                  + '；请核对该处语境并在这个数值所在句子后引用对应编号，不要只在整段末尾引用上一页。')
    return issues


def support_issues(text, context, evidence, question=""):
    """Check each sentence against its own references; retain explicit limits."""
    text = normalize_math_text(text)
    issues = calculation_issues(text) + percentage_issues(text, question)
    if (re.search(r'图\s*\d+', question) and re.search(r'百分点|差值|增加多少|减少多少|相差', question)
            and re.search(r'\d', text) and not re.search(r'没有足够依据|无法确认|无法确定', text)
            and not re.search(r'\d+(?:\.\d+)?\s*[%％]?\s*[-−]\s*\d+(?:\.\d+)?\s*[%％]?\s*[=＝]', text)):
        issues.append('图表差值回答缺少原始数值和减法计算式；请先引用两项原始读数及单位，再计算。看不清则明确说明无法确认。')
    # A reference immediately after terminal punctuation belongs to that sentence.
    sentences = re.findall(r'[^。！？；;\n]+(?:[。！？；;](?:\s*\[E\d+\])*)?', text)
    for sentence in sentences:
        refusal = re.search(r'没有足够依据|无法确认|无法确定|无法判断|不能确认|无法读取|未提供|不能作为', sentence)
        measurement = re.search(r'\d+(?:\.\d+)?\s*(?:[%％]|个百分点|万吨|万元|亿元|万人|吨|元|人)', sentence)
        if refusal and not measurement:
            # Mentioning the requested year/figure while explaining missing
            # evidence is not a numerical answer. Do not exempt real readings.
            continue
        refs = set(re.findall(r'\[(E\d+)\]', sentence))
        selected = [c for c in context if c['label'] in refs]
        if not refs and re.search(r'\d+\.\d+', sentence):
            issues.append('包含数值的句子缺少就近引用：' + sentence.strip()[:100])
        figure = re.search(r'图\s*(\d+)', question)
        if figure and numeric_values(re.sub(r'\[E\d+\]', '', sentence)):
            matching = []
            wrong = []
            for i, e in enumerate(evidence, 1):
                label = f'E{i}'
                heading = re.match(r'^图\s*(\d+)', e.element.text.strip())
                if heading and heading[1] == figure[1] and any(c['label'] == label and c['image_sent'] for c in context):
                    matching.append(label)
                if heading and heading[1] != figure[1] and label in refs and e.element.kind == 'image':
                    wrong.append(label)
            if wrong and not any(label in refs for label in matching):
                hint = '；对应图像为' + '、'.join(f'[{x}]' for x in matching) if matching else ''
                issues.append('该句引用图号与问题图' + figure[1] + '不一致：' + '、'.join(wrong) + hint)
        issues.extend(_fragment_issues(sentence, context, evidence))
        issues.extend(table_claim_issues(sentence, selected))
    if re.search(r'分别|各自', question):
        rows = {re.sub(r'\[\d+\]|其中[:：]', '', row) for c in context
                for row in re.findall(r'^行=(.*?)；列=', c['text'], re.M)}
        requested = {row for row in rows if row and row in question}
        if len(requested) >= 2:
            missing = sorted(row for row in requested if row not in text)
            if missing:
                issues.append('回答遗漏问题明确要求的项目：' + '、'.join(missing)
                              + '；请分别作答并引用，无法确认的项目也需明确说明。')
    return list(dict.fromkeys(issues))
