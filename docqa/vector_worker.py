"""FAISS runs outside the model process so its bundled OpenMP runtime stays isolated."""

import json
import sys
from pathlib import Path

import faiss
import numpy as np


def main():
    request = json.load(sys.stdin)
    path = Path(request["vectors_path"])
    index_path = path.with_suffix(".faiss")
    faiss.omp_set_num_threads(1)
    if index_path.exists():
        index = faiss.read_index(str(index_path))
    else:
        with np.load(path, allow_pickle=False) as data:
            vectors = np.ascontiguousarray(data["vectors"], dtype="float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(index_path))
    query = np.ascontiguousarray(request["query"], dtype="float32")
    scores, ids = index.search(query, min(request["k"], index.ntotal))
    json.dump({"scores": scores[0].tolist(), "indices": ids[0].tolist()}, sys.stdout, allow_nan=False)


if __name__ == "__main__":
    main()
