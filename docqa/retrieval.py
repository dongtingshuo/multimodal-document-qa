import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .evidence_quality import page_continuations
from .schemas import Chunk, Evidence
from .structure import OUTLINE_VERSION, restore_section_paths

RETRIEVAL_VERSION = "adjacent-table-and-text-v6"


def table_query_hint(question):
    """Detect explicit table-oriented wording without classifying every short query as a table query."""

    return bool(
        re.search(
            r"表格|数据表|下表|表中|浓度限值|限值|平均时间|污染物项目|一级|二级|表[：:]",
            question,
        )
    )


def named_document_scope(question, documents):
    """Resolve exact, explicitly quoted document titles only within the allowed corpus."""

    def normalized_title(text):
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()

    titles = set()
    for match in re.finditer(r"(?P<prefix>第\s*\d+\s*次)?\s*《(?P<title>[^《》]+)》", question):
        prefix = match.group("prefix") or ""
        titles.add(normalized_title(prefix + match.group("title")))
    matches = [d["id"] for d in documents if normalized_title(Path(d["name"]).stem) in titles]
    return matches or None


def vector_search(path, query, k):
    env = dict(os.environ)
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    result = subprocess.run(
        [sys.executable, "-m", "docqa.vector_worker"],
        input=json.dumps({"vectors_path": str(path.resolve()), "query": query.tolist(), "k": k}),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
        env=env,
    )
    if result.returncode:
        raise RuntimeError("FAISS 检索进程失败：" + result.stderr[-300:])
    output = json.loads(result.stdout)
    return [(int(i), float(s)) for i, s in zip(output["indices"], output["scores"], strict=True) if i >= 0]


def tokenize(text):
    return [t.lower() for t in jieba.lcut(text) if re.search(r"[\w\u4e00-\u9fff]", t)]


def chunks_for(elements, limit=400, overlap=60, tokenizer=None):
    """Preserve source substrings; use model token offsets when the tokenizer is available."""

    def offsets(text):
        if tokenizer is None:
            return [(i, i + 1) for i in range(len(text))]
        return tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)[
            "offset_mapping"
        ]

    def windows(text, budget=limit, shared=overlap):
        spans = offsets(text)
        result = []
        for start in range(0, len(spans), max(1, budget - shared)):
            end = min(start + budget, len(spans))
            result.append(text[spans[start][0] : spans[end - 1][1]])
            if end == len(spans):
                break
        return result

    chunks = []
    for e in elements:
        parts = []
        if e.kind == "table" and e.table:
            # Parser text may include the table caption before its Markdown grid.
            # Repeat it alongside the column headers in each row group.
            caption = e.text.partition("|")[0].strip() if "|" in e.text else ""
            if len(offsets(caption)) > limit // 4:
                caption = ""  # Never repeat a full text body as a caption.
            header = (caption + "\n" if caption else "") + " | ".join(e.table[0])
            pending = header
            for row in e.table[1:]:
                line = " | ".join(row)
                if len(offsets(pending + "\n" + line)) > limit and pending != header:
                    parts.append(pending)
                    pending = header
                pending += "\n" + line
            parts.append(pending)
        else:
            parts = windows(e.text)
        bounded = []
        for part in parts:
            if len(offsets(part)) <= limit:
                bounded.append(part)
            elif e.kind == "table" and len(offsets(header)) < limit // 2:
                body = part[len(header) :].lstrip("\n")
                bounded.extend(header + "\n" + p for p in windows(body, limit - len(offsets(header)) - 4, 0))
            else:
                bounded.extend(windows(part))
        for i, text in enumerate(bounded):
            if text.strip():
                chunks.append(Chunk(id=f"{e.id}-c{i}", element_id=e.id, document_id=e.document_id, text=text))
    return chunks


class ModelRegistry:
    def __init__(self, settings):
        self.settings = settings
        self._embedding = self._reranker = self._clip = None
        self.lock = threading.RLock()

    def embedding(self):
        with self.lock:
            if self._embedding is None:
                from sentence_transformers import SentenceTransformer

                self._embedding = SentenceTransformer(
                    self.settings.embedding_model,
                    device=self.settings.device,
                    local_files_only=self.settings.local_models_only,
                )
            return self._embedding

    def reranker(self):
        with self.lock:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(
                    self.settings.reranker_model,
                    device=self.settings.device,
                    local_files_only=self.settings.local_models_only,
                    max_length=512,
                )
            return self._reranker

    def clip(self):
        with self.lock:
            if not self.settings.enable_clip:
                raise RuntimeError("Chinese-CLIP 未开启，请配置 DOCQA_ENABLE_CLIP=true 并准备模型")
            if self._clip is None:
                from cn_clip.clip import load_from_name

                self._clip = load_from_name(
                    self.settings.clip_model,
                    device=self.settings.device,
                    download_root=str(self.settings.clip_model_dir),
                )
                self._clip[0].eval()
            return self._clip


def rrf(rankings, constant=60):
    fused = {}
    for ranking in rankings:
        for rank, eid in enumerate(dict.fromkeys(ranking), 1):
            fused[eid] = fused.get(eid, 0.0) + 1 / (constant + rank)
    return fused


class HybridRetriever:
    def __init__(self, store, settings, models=None):
        self.store, self.settings = store, settings
        self.models = models or ModelRegistry(settings)
        self.cache = {}
        self.lock = threading.RLock()
        self.tokenizer = None
        self.tokenizer_checked = False
        self.chunk_cache = None
        self.bm25_cache = None

    def chunks(self, elements):
        if not self.tokenizer_checked:
            from transformers import AutoTokenizer

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.embedding_model, local_files_only=True
                )
            except OSError:
                # BM25 can run before model download. Trace explicitly identifies this fallback.
                self.tokenizer = None
            self.tokenizer_checked = True
        names = {d["id"]: d["name"] for d in self.store.documents()}
        signature = hashlib.sha256(
            json.dumps(
                [(e.id, e.text, e.table, e.title_path, names.get(e.document_id)) for e in elements],
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        if self.chunk_cache and self.chunk_cache[0] == signature:
            return self.chunk_cache[1]
        chunks = chunks_for(elements, tokenizer=self.tokenizer)
        by_id = {e.id: e for e in elements}
        for chunk in chunks:
            element = by_id[chunk.element_id]
            heading = " / ".join(element.title_path)[-160:]
            prefix = f"文档：{names.get(element.document_id, '')[:120]}"
            if heading:
                prefix += f"\n章节：{heading}"
            if self.tokenizer is not None:
                offsets = self.tokenizer(prefix, add_special_tokens=False, return_offsets_mapping=True)[
                    "offset_mapping"
                ]
                if len(offsets) > 80:
                    prefix = prefix[: offsets[79][1]]
            else:
                prefix = prefix[:80]
            chunk.text = prefix + "\n" + chunk.text
        self.chunk_cache = (signature, chunks)
        return chunks

    def invalidate(self):
        with self.lock:
            self.cache.clear()
            self.chunk_cache = self.bm25_cache = None
            for suffix in ("*.npz", "*.faiss"):
                for path in (self.settings.data_dir / "indexes").glob(suffix):
                    path.unlink(missing_ok=True)

    def dense_scores(self, query, chunks):
        model = self.models.embedding()
        signature = hashlib.sha256(
            json.dumps(
                [self.settings.embedding_model, [(c.id, c.text) for c in chunks]], ensure_ascii=False
            ).encode()
        ).hexdigest()
        cache_dir = self.settings.data_dir / "indexes"
        cache_dir.mkdir(exist_ok=True)
        path = cache_dir / f"{signature}.npz"
        with self.lock:
            if signature not in self.cache:
                if path.exists():
                    with np.load(path, allow_pickle=False) as data:
                        vectors = data["vectors"]
                else:
                    vectors = model.encode(
                        [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False
                    )
                    np.savez_compressed(path, vectors=vectors)
                # Bound in-memory indexes across changing document scopes.
                self.cache.clear()
                self.cache[signature] = path
            prompt = (
                "为这个句子生成表示以用于检索相关文章："
                if "bge-small-zh" in self.settings.embedding_model
                else ""
            )
            vector = model.encode([prompt + query], normalize_embeddings=True, show_progress_bar=False)
            return vector_search(path, np.asarray(vector, dtype="float32"), len(chunks))

    def visual_scores(self, query, elements):
        import torch
        from cn_clip.clip import tokenize as clip_tokenize
        from PIL import Image

        model, preprocess = self.models.clip()
        visual = [e for e in elements if e.image_path and e.kind in {"image", "page", "table"}]
        if not visual:
            return []
        signature = hashlib.sha256(
            (self.settings.clip_model + "|".join(e.id for e in visual)).encode()
        ).hexdigest()
        directory = self.settings.data_dir / "indexes"
        directory.mkdir(exist_ok=True)
        path = directory / f"clip-{signature}.npz"
        with self.models.lock, torch.no_grad():
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    vectors = data["vectors"]
            else:
                values = []
                for e in visual:
                    with Image.open(e.image_path) as img:
                        value = model.encode_image(
                            preprocess(img.convert("RGB")).unsqueeze(0).to(self.settings.device)
                        )
                        values.append((value / value.norm(dim=-1, keepdim=True)).cpu().float().numpy()[0])
                vectors = np.asarray(values, dtype="float32")
                np.savez_compressed(path, vectors=vectors)
            q = model.encode_text(clip_tokenize([query]).to(self.settings.device))
            q = (q / q.norm(dim=-1, keepdim=True)).cpu().float().numpy()
        return [(visual[i].id, score) for i, score in vector_search(path, q, min(20, len(visual)))]

    def retrieve(self, request):
        start = time.perf_counter()
        documents = [
            d
            for d in self.store.documents()
            if d["status"] == "ready" and (request.document_ids is None or d["id"] in request.document_ids)
        ]
        named_scope = named_document_scope(request.question, documents)
        elements = restore_section_paths(
            self.store.elements(named_scope if named_scope is not None else request.document_ids)
        )
        if request.kinds is not None:
            elements = [e for e in elements if e.kind in request.kinds]
        by_id = {e.id: e for e in elements}
        chunks = self.chunks(elements)
        rankings, scores, matches = {}, {}, {}

        def aggregate(channel, values):
            ranking = []
            for i, score in values:
                chunk = chunks[i]
                eid = chunk.element_id
                if eid in ranking:
                    continue
                ranking.append(eid)
                scores.setdefault(eid, {})[channel] = score
                matches.setdefault(eid, chunk.text)
                if len(ranking) == 30:
                    break
            rankings[channel] = ranking

        if chunks and request.strategy != "dense":
            signature = hashlib.sha256(
                json.dumps([(c.id, c.text) for c in chunks], ensure_ascii=False).encode()
            ).hexdigest()
            if not self.bm25_cache or self.bm25_cache[0] != signature:
                corpus = [tokenize(c.text) or ["__empty__"] for c in chunks]
                bm = BM25Okapi(corpus)
                # Count document frequency once, then reuse the index for this scope.
                frequencies = Counter(term for doc in bm.doc_freqs for term in doc)
                for term, df in frequencies.items():
                    bm.idf[term] = float(np.log(1 + (len(corpus) - df + 0.5) / (df + 0.5)))
                self.bm25_cache = (signature, bm)
            bm = self.bm25_cache[1]
            values = bm.get_scores(tokenize(request.question))
            aggregate(
                "bm25",
                [(int(i), float(values[i])) for i in np.argsort(-values, kind="stable") if values[i] > 0],
            )
        if chunks and request.strategy != "bm25":
            aggregate("dense", self.dense_scores(request.question, chunks))
        if request.multimodal:
            visual = self.visual_scores(request.question, elements)
            rankings["clip"] = [eid for eid, _ in visual]
            for eid, score in visual:
                scores.setdefault(eid, {})["clip"] = score
                matches.setdefault(eid, by_id[eid].text)
        fused = rrf(list(rankings.values()))
        ordered = sorted(fused, key=lambda eid: (-fused[eid], eid))
        for eid in ordered:
            scores[eid]["rrf"] = fused[eid]
        fusion_order = ordered.copy()
        rerank_start = time.perf_counter()
        if request.strategy == "hybrid_reranker" and ordered:
            candidates = ordered[:30]
            values = self.models.reranker().predict([(request.question, matches[eid]) for eid in candidates])
            for eid, value in zip(candidates, values, strict=False):
                scores[eid]["reranker"] = float(value)
            ordered = sorted(candidates, key=lambda eid: -scores[eid]["reranker"])
            # Cross-encoder scores can saturate on long Markdown tables. If the
            # question explicitly requests a table, keep the best table candidate
            # visible before text snippets that answer a different statistic.
            if table_query_hint(request.question):
                table_candidates = [eid for eid in candidates if by_id[eid].kind == "table"]
                if table_candidates:
                    best_table = max(
                        table_candidates,
                        key=lambda eid: (scores[eid]["reranker"], scores[eid]["rrf"]),
                    )
                    ordered = [best_table] + [eid for eid in ordered if eid != best_table]
        continuations = []
        expanded = []
        for eid in ordered:
            if eid not in expanded:
                expanded.append(eid)
            if len(expanded) < request.top_k and len(continuations) < 2:
                pending = [eid]
                while pending and len(continuations) < 2 and len(expanded) < request.top_k:
                    anchor_id = pending.pop(0)
                    for neighbor in page_continuations(by_id[anchor_id], elements):
                        if neighbor.id in expanded:
                            continue
                        if len(continuations) >= 2 or len(expanded) >= request.top_k:
                            break
                        expanded.append(neighbor.id)
                        pending.append(neighbor.id)
                        continuations.append({"anchor": anchor_id, "element_id": neighbor.id})
                        matches.setdefault(neighbor.id, neighbor.text)
                        scores.setdefault(neighbor.id, {})["page_continuation"] = 1.0
        ordered = expanded
        if continuations:
            rankings["page_continuation"] = [c["element_id"] for c in continuations]
        docs = {d["id"]: d for d in self.store.documents()}
        evidence = [
            Evidence(
                element=by_id[eid],
                document_name=docs[by_id[eid].document_id]["name"],
                page_label="转换版页码" if docs[by_id[eid].document_id].get("converted") else "页码",
                matched_text=matches[eid],
                rank=i,
                channels=[c for c in rankings if eid in rankings[c]],
                scores=scores[eid],
            )
            for i, eid in enumerate(ordered[: request.top_k], 1)
        ]
        return evidence, {
            "rankings": rankings,
            "fusion": fusion_order,
            "final": ordered,
            "timings": {
                "retrieval_seconds": time.perf_counter() - start,
                "rerank_seconds": time.perf_counter() - rerank_start,
            },
            "embedding_model": self.settings.embedding_model,
            "reranker_model": self.settings.reranker_model,
            "chunking": "400 content tokens, 60 overlap, plus <=80 document/section tokens"
            if self.tokenizer is not None
            else "400 characters, 60 overlap (tokenizer unavailable)",
            "bm25_idf": "log(1 + (N-df+0.5)/(df+0.5))",
            "rrf_constant": 60,
            "evidence_enrichment": OUTLINE_VERSION,
            "retrieval_version": RETRIEVAL_VERSION,
            "named_document_scope": named_scope,
            "table_query_hint": table_query_hint(request.question),
            "page_continuations": continuations,
        }
