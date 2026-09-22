"""Archive 20 real API highlight previews for separate visual review."""
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from docqa.api import create_app
from docqa.config import Settings
from docqa.store import Store


def main():
    out = ROOT / 'data/completion-20260911/highlights'
    out.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    store = Store(settings.data_dir)
    app = create_app(settings)
    # Exercise actual API routes against existing read-only records, without
    # starting a second service worker or modifying document lifecycle states.
    app.state.service = SimpleNamespace(store=store)
    client = TestClient(app)
    questions = json.loads((ROOT / 'datasets/public_questions.json').read_text())['questions']
    selected, seen = [], set()
    for category, count in [('text', 8), ('table', 6), ('chart', 6)]:
        found = 0
        for q in questions:
            if q['category'] != category:
                continue
            eid = q['relevant_element_ids'][0]
            if eid in seen:
                continue
            selected.append(q)
            seen.add(eid)
            found += 1
            if found == count:
                break
    assert len(selected) == 20
    rows, tiles = [], []
    for q in selected:
        eid = q['relevant_element_ids'][0]
        response = client.get('/api/v1/evidence/' + eid)
        response.raise_for_status()
        element = response.json()
        src = element['sources'][0]
        did, page = element['document_id'], src['page']
        doc = store.document(did)
        response = client.get(f'/api/v1/documents/{did}/pages/{page}', params={'highlight': eid})
        response.raise_for_status()
        path = out / (q['id'] + '.png')
        path.write_bytes(response.content)
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        w, h = img.size
        x0,y0,x1,y1 = src['bbox']
        region = img.crop((max(0,int(x0*w)-12), max(0,int(y0*h)-12),
                           min(w,int(x1*w)+12), min(h,int(y1*h)+12)))
        tile = Image.new('RGB', (1450, 610), 'white')
        thumb = img.copy()
        thumb.thumbnail((360,550))
        tile.paste(thumb,(10,45))
        region.thumbnail((1060,550))
        tile.paste(region,(380,45))
        ImageDraw.Draw(tile).text((12,12), f"{q['id']} | {q['category']} | PDF page {page} | {eid}",fill='black')
        tiles.append(tile)
        rows.append({'question_id':q['id'],'question':q['question'], 'reference_answer':q['reference_answer'],
                     'document':doc['name'], 'element_id':eid, 'page':page, 'bbox':src['bbox'],
                     'preview':path.name,'sha256':hashlib.sha256(response.content).hexdigest(),
                     'api_status':response.status_code,'visual_review':'pending'})
    for index in range(5):
        sheet = Image.new('RGB',(1450,2440),'white')
        for j, tile in enumerate(tiles[index*4:index*4+4]):
            sheet.paste(tile,(0,j*610))
        sheet.save(out/f'sheet-{index+1}.png')
    (out/'cases.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print('20 API previews and 5 visual sheets archived')


if __name__ == '__main__':
    main()
