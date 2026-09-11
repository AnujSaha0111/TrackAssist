# Frozen hybrid intent classifier (word+char TF-IDF + MiniLM + LogReg).
# Loads the shipped TF-IDF union and logistic regression; MiniLM vectors come from src.retrieval (shared frozen encoder). No fitting happens here

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from . import leakage as lg
from . import text as T

ROOT = Path(__file__).resolve().parent.parent
B_UNION = ROOT / "artifacts" / "models" / "b_tfidf_union.joblib"
HYBRID_CLF = ROOT / "artifacts" / "models" / "hybrid_classifier.joblib"

_state = {}


def load():
    if _state:
        return _state
    lg.assert_taxonomy_unchanged()
    from .retrieval import encode_texts
    _state.update(b_union=joblib.load(B_UNION),
                  clf=joblib.load(HYBRID_CLF), encode=encode_texts)
    return _state


def predict(text, prior_texts=None):
    """Returns (intent, confidence). Customer text only."""
    s = load()
    row = {"dyad_id": "QUERY", "target_text": text,
           "prior_customer_context": list(prior_texts or []), "context": []}
    X_lex = s["b_union"].transform([T.joined_text(row)])
    X_sem = s["encode"]([T.semantic_text(row)])
    X = hstack([X_lex, csr_matrix(X_sem)], format="csr")
    proba = s["clf"].predict_proba(X)[0]
    idx = int(np.argmax(proba))
    return str(s["clf"].classes_[idx]), round(float(proba[idx]), 4)