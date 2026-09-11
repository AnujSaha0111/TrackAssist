# Historical resolution retrieval (frozen MiniLM encoder + cosine top-k)
# Corpus: data/corpus.jsonl (3,985 non-golden dyads with brand turns)
# Query: customer text only. Exclusions per query: own dyad, same customer, normalized-text duplicates. Golden dyads are refused at load time
# First run downloads the ~90MB MiniLM weights from Hugging Face (network needed once); afterwards everything runs offline on CPU

import json
from pathlib import Path

import numpy as np
import torch

from . import leakage as lg
from . import text as T

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus.jsonl"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 64
SEED = 42
TOPK = 10

_model = None
_index = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        torch.manual_seed(SEED)
        _model = SentenceTransformer(MODEL_ID, device="cpu")
        _model.eval()
    return _model


def encode_texts(texts, batch_size=BATCH_SIZE):
    texts = list(texts)
    assert all(isinstance(t, str) and t.strip() for t in texts)
    model = get_model()
    with torch.no_grad():
        vecs = model.encode(texts, batch_size=batch_size,
                            show_progress_bar=False,
                            normalize_embeddings=True)
    arr = np.asarray(vecs, dtype=np.float32)
    assert arr.shape == (len(texts), EMBEDDING_DIM)
    assert np.all(np.isfinite(arr))
    return arr


def load_corpus():
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    lg.assert_no_golden_dyads([r["dyad_id"] for r in rows], "retrieval corpus")
    return rows


def build_index(rows=None):
    """Encode the corpus once per process; returns (docs, matrix)."""
    global _index
    if _index is not None:
        return _index
    docs = rows or load_corpus()
    X = encode_texts([T.semantic_text(r) for r in docs])
    _index = (docs, X)
    return _index


def retrieve(query_text, prior_texts=None, query_dyad_id="QUERY",
             query_customer_id="LIVE", k=TOPK):
    """Top-k dyads excluding own dyad, same customer, and text duplicates."""
    docs, X = build_index()
    row = {"dyad_id": query_dyad_id, "target_text": query_text,
           "prior_customer_context": list(prior_texts or []), "context": []}
    qv = encode_texts([T.semantic_text(row)])
    sim = (qv @ X.T)[0]
    order = np.argsort(-sim, kind="stable")[:k * 3]
    qnorm = T.norm_text(query_text)
    out = []
    for di in order:
        d = docs[int(di)]
        if d["dyad_id"] == query_dyad_id:
            continue
        if str(d["customer_id"]) == str(query_customer_id):
            continue
        if T.norm_text(d["target_text"]) == qnorm:
            continue
        out.append({"dyad_id": d["dyad_id"], "similarity": round(float(sim[di]), 4),
                    "doc": d})
        if len(out) >= k:
            break
    return out
