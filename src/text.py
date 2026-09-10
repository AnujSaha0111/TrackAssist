# Customer-text input builders (customer text only, never brand text).
# Two frozen input forms, matching how each representation was fitted:
# - joined_text: target + prior context, single-space join (TF-IDF branches)
# - semantic_text: target + prior context with a clear ' || ' separator (MiniLM branch)

import html
import re

SEPARATOR = " || "
FORBIDDEN_ROW_KEYS = {"brand_responses", "brand_responses_substantive",
                      "full_thread", "messages", "response_tweet_id",
                      "author_type", "is_substantive"}

def dec(t):
    return html.unescape(t or "")

def norm_text(t):
    return re.sub(r"\s+", " ", (t or "").lower()).strip()

def assert_customer_row(row):
    if not isinstance(row, dict):
        raise AssertionError("row must be a mapping")
    bad = FORBIDDEN_ROW_KEYS & set(row)
    if bad:
        raise AssertionError(f"refused row with forbidden keys: {sorted(bad)}")
    for key in ("prior_customer_context", "context"):
        for m in row.get(key) or []:
            if isinstance(m, dict) and \
                    m.get("author_type", "customer") != "customer":
                raise AssertionError("non-customer turn in permitted context")
    return True

def context_texts_of(row):
    ctx = []
    for key in ("prior_customer_context", "context"):
        for m in row.get(key) or []:
            ctx.append(m.get("text", "") if isinstance(m, dict) else m)
    seen, uniq = set(), []
    for c in ctx:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return [dec(c) for c in uniq if dec(c).strip()]

def joined_text(row):
    parts = [dec(row.get("target_text", ""))] + context_texts_of(row)
    return " ".join(p for p in parts if p).strip()

def semantic_text(row):
    assert_customer_row(row)
    target = dec(row.get("target_text", "")).strip()
    if not target:
        raise AssertionError("empty target text")
    ctx = context_texts_of(row)
    return target if not ctx else target + SEPARATOR + SEPARATOR.join(ctx)