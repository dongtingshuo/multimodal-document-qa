"""Conservative adjacent-page links; never rewrite original table cells/sources."""
import re


def _heading(text):
    return bool(re.match(r'^(?:第[一二三四五六七八九十\d]+[章节]|[一二三四五六七八九十]+、|[（(][一二三四五六七八九十]+[）)])', text.strip()))


def table_neighbors(anchor, elements):
    if anchor.kind != 'table' or not anchor.table or len(anchor.sources) != 1:
        return []
    pos = anchor.sources[0]
    headers = [str(c).strip() for c in anchor.table[0]]
    if len(headers) < 2 or any(not h for h in headers) or len(set(headers)) != len(headers):
        return []
    def number(element):
        match = re.search(r'(?:^|\n)\s*(?:续)?表\s*(\d+)', element.text)
        return match[1] if match else None
    linked = []
    for other in elements:
        if other.id == anchor.id or other.document_id != anchor.document_id or other.kind != 'table':
            continue
        if not other.table or len(other.sources) != 1:
            continue
        nxt = other.sources[0]
        if abs(nxt.page - pos.page) != 1 or headers != [str(c).strip() for c in other.table[0]]:
            continue
        first, second = (anchor, other) if pos.page < nxt.page else (other, anchor)
        bottom, top = first.sources[0], second.sources[0]
        if bottom.bbox[3] < .75 or top.bbox[1] > .25:
            continue
        if abs(bottom.bbox[0] - top.bbox[0]) > .08 or abs(bottom.bbox[2] - top.bbox[2]) > .08:
            continue
        if first.title_path != second.title_path:
            continue
        units = [re.search(r'单位[:：]\s*([^\n，,；;]+)', e.text) for e in (first, second)]
        if all(units) and units[0][1].strip() != units[1][1].strip():
            continue
        a, b = number(first), number(second)
        # Same headers alone are not enough: unrelated tables often share them.
        if not ((a and b and a == b) or (a and not b and '续表' in second.text[:40])):
            continue
        if any(e.kind == 'text' and e.document_id == anchor.document_id
               and len(e.sources) == 1 and e.sources[0].page == top.page
               and e.sources[0].bbox[1] < top.bbox[1] and _heading(e.text) for e in elements):
            continue
        linked.append(other)
    # An ambiguous choice on one page must not create a spurious link.
    return [e for e in linked if sum(x.sources[0].page == e.sources[0].page for x in linked) == 1]
