"""Conservative table normalization and exact, non-executing arithmetic checks."""
import re
from decimal import Decimal

NUMBER = r'[-+]?\d+(?:\.\d+)?'


def table_cells(table):
    """Accept rectangular tables; unpack collapsed rows only on exact alignment."""
    if not table or len(table) < 2 or len(table[0]) < 2:
        return []
    width = len(table[0])
    if width > 20 or any(len(row) != width for row in table):
        return []
    rows = [[str(c).strip() for c in row] for row in table]
    headers = rows[0][1:]
    start = 1
    if len(rows) > 2 and not rows[1][0]:
        # Some parsers combine the column headings in one cell while keeping
        # the following header row partially aligned. Never guess missing names.
        tokens = headers[0].split() if headers else []
        if len(tokens) == width - 1 and all(not c for c in headers[1:]):
            if any(c and c != tokens[i] for i, c in enumerate(rows[1][1:])):
                return []
            headers = tokens
        else:
            headers = [b or a for a, b in zip(headers, rows[1][1:], strict=True)]
        start = 2
    if any(not h for h in headers) or len(set(headers)) != len(headers):
        return []
    cells = []
    for row in rows[start:]:
        names = [row[0]]
        columns = [[c] for c in row[1:]]
        if not all(re.fullmatch(NUMBER + r'[%％]?', c) for c in row[1:]):
            split = [c.split() for c in row[1:]]
            labels = row[0].split()
            if len(labels) < 2 or not all(len(c) == len(labels) for c in split):
                continue
            if not all(re.fullmatch(NUMBER + r'[%％]?', v) for c in split for v in c):
                continue
            names, columns = labels, split
        for j, name in enumerate(names):
            if not name:
                continue
            for header, column in zip(headers, columns, strict=True):
                if len(cells) >= 400:
                    return []
                cells.append({'row': name, 'column': header, 'value': column[j]})
    return cells


def table_notes(table, question):
    cells = table_cells(table)
    requested = [c for c in cells if re.sub(r'\[\d+\]|其中[:：]', '', c['row']) in question]
    cells = requested if requested else cells[:24]
    lines = [f"行={c['row']}；列={c['column']}；数值={c['value']}" for c in cells]
    # Only compute explicitly named chronological endpoints in one table row.
    years = re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)', question)
    if len(set(years)) == 2 and re.search(r'增加|增长|增幅|相对|减少|提高|下降|变化|差|百分点', question):
        early, late = sorted(set(years))
        for row in dict.fromkeys(c['row'] for c in requested):
            pair = []
            for year in (early, late):
                matches = [c for c in requested if c['row'] == row and re.fullmatch(year + r'年?', c['column'])]
                if len(matches) != 1:
                    break
                pair.append(matches[0]['value'])
            if len(pair) == 2 and pair[0].endswith(('%', '％')) == pair[1].endswith(('%', '％')):
                a, b = (Decimal(v.rstrip('%％')) for v in pair)
                unit = '个百分点' if pair[0].endswith(('%', '％')) else '（沿用原表单位；未标明则不能推断）'
                lines.append(f'程序计算：{row}，{late}年减{early}年：{b} - {a} = {b-a}{unit}')
                if re.search(r'增长率|相对|增幅|增长百分之', question) and '百分点' not in question:
                    if a == 0:
                        lines.append('相对增长率：基期为零，不能计算通常定义的增长率。')
                    else:
                        rate = ((b-a)/a*100).quantize(Decimal('0.0001'))
                        lines.append(f'程序计算：相对增长率=({b}-{a})/{a}×100%，约为{rate}%；与百分点差值不同。')
    return '\n'.join(lines)


def normalize_math_text(text):
    """Normalize display-only math escapes for checks, without changing the answer."""
    return text.replace(r"\%", "%").replace(r"\[", "").replace(r"\]", "").replace(r"\(", "").replace(r"\)", "")


def calculation_issues(text):
    # Parse only literal arithmetic. Never eval model or document content.
    clean = re.sub(r'\[E\d+\]|\*\*', '', normalize_math_text(text)).replace('−', '-').replace('＝', '=')
    issues = []
    for match in re.finditer(rf'({NUMBER})\s*([+\-×*÷/])\s*({NUMBER})\s*=\s*({NUMBER})', clean):
        a, op, b, result = match.groups()
        a, b, result = Decimal(a), Decimal(b), Decimal(result)
        if op in '/÷' and b == 0:
            issues.append('算式除数为零，不能给出有限结果。')
            continue
        expected = a+b if op == '+' else a-b if op == '-' else a*b if op in '*×' else a/b
        # Explicit approximate rounding should use ≈, not equality.
        if result != expected:
            issues.append(f'算式不成立：{match[0]}；程序计算结果为{expected}。请核对原始数值和单位。')
    return issues


def table_claim_issues(sentence, selected):
    issues = []
    for context in selected:
        cells = re.findall(r'^行=(.*?)；列=(.*?)；数值=([^\n]+)$', context['text'], re.M)
        row_names = {re.sub(r'\[\d+\]|其中[:：]', '', r) for r, _, _ in cells}
        for row, column, value in cells:
            row_name = re.sub(r'\[\d+\]|其中[:：]', '', row)
            if row_name not in sentence:
                continue
            # Bind a value directly to its named column; don't compare against
            # unrelated numbers from another column or sentence.
            pattern = re.escape(column) + rf'(?:的)?(?:变化|价格|涨跌幅度|数值)?(?:为|是)?\s*(下降|降低|减少|增长|上升)?\s*({NUMBER})'
            for match in re.finditer(pattern, sentence):
                # Bind the column to the nearest preceding explicit row name.
                # A table title may itself contain the name of a total row.
                preceding = sentence[:match.start()]
                nearest = max((name for name in row_names if name in preceding),
                              key=lambda name: preceding.rfind(name), default=None)
                if nearest != row_name:
                    continue
                direction, number = match.groups()
                observed = Decimal(number)
                if direction in {'下降', '降低', '减少'}:
                    observed = -abs(observed)
                expected = Decimal(value.rstrip('%％'))
                if observed != expected:
                    issues.append(f"[{context['label']}]的{row}／{column}为{value}，与该句{number}不符；请检查行列。")
    return issues


def percentage_issues(text, question=''):
    """Distinguish percent-point differences from relative percent growth."""
    clean = re.sub(r'\[E\d+\]|\*\*', '', normalize_math_text(text)).replace('−', '-').replace('＝', '=')
    issues = []
    for m in re.finditer(rf'({NUMBER})\s*[%％]\s*-\s*({NUMBER})\s*[%％]\s*=\s*({NUMBER})\s*(个百分点|[%％])', clean):
        a, b, result, unit = m.groups()
        expected = Decimal(a) - Decimal(b)
        if Decimal(result) != expected:
            issues.append(f'百分数差值算式错误，应为{expected}个百分点。')
        if unit != '个百分点' and '百分点' in question:
            issues.append('问题问的是百分点差值，请把百分数之差明确表述为“个百分点”，不要与相对增长百分比混用。')
    if '百分点' in question:
        for sentence in re.split(r'[。；;\n]', clean):
            if re.search(r'(?:提高|增加|下降|减少|相差)(?:了)?\s*' + NUMBER + r'\s*[%％]', sentence) and not re.search(r'相对|增长率|增幅', sentence):
                issues.append('该句把百分点变化写成了百分比，请明确区分百分点差值与相对增长率。')
    return issues
