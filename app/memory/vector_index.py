import os
import faiss
import numpy as np

class FaissIndex:
    def __init__(self, faiss_path: str, dim: int):
        self.faiss_path = faiss_path
        self.dim = dim

        if os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
        else:
            base = faiss.IndexFlatIP(dim)         # cosine if vectors normalized
            self.index = faiss.IndexIDMap2(base)  # lets us store with event_id

    def add(self, vec: np.ndarray, event_id: int):
        v = vec.reshape(1, -1).astype("float32")
        self.index.add_with_ids(v, np.array([event_id], dtype=np.int64))

    def search(self, vec: np.ndarray, k: int):
        if self.index.ntotal == 0:
            return [], []
        v = vec.reshape(1, -1).astype("float32")
        D, I = self.index.search(v, min(k, self.index.ntotal))
        scores = [float(x) for x in D[0] if x != -1]
        ids = [int(i) for i in I[0] if i != -1]
        return scores, ids

    def save(self):
        faiss.write_index(self.index, self.faiss_path)
