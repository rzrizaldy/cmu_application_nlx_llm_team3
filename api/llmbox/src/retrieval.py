"""
retrieval.py  [A2 M5]

Retrieval few-shot (batch.submode=fewshot). For each input, find the k most
similar *labelled* records in a pool (TF-IDF cosine, pure Python so it runs on
the VM without scikit-learn) and show them to the model as worked examples
(user: record, assistant: its JSON tags) before the real input.

Leakage rules: the pool file must only contain DEV records; a row is never
given itself as an example; every retrieved id is returned so the run can be
audited against the EVAL split.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

_WORD = re.compile(r"[a-z0-9]+")


def _terms(text):
    return _WORD.findall(text.lower())


class TfidfIndex:
    def __init__(self, docs):
        """docs: list of (id, text)."""
        self.ids = [d for d, _ in docs]
        tfs = [Counter(_terms(t)) for _, t in docs]
        df = Counter(term for tf in tfs for term in tf)
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vecs = [self._vec(tf) for tf in tfs]

    def _vec(self, tf):
        v = {t: c * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, text, k, exclude=()):
        q = self._vec(Counter(_terms(text)))
        scored = [(sum(w * vec.get(t, 0.0) for t, w in q.items()), i) for i, vec in zip(self.ids, self.vecs)
                  if i not in exclude]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(i, round(s, 4)) for s, i in scored[:k]]


class FewShotPool:
    def __init__(self, path):
        rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        self.rows = {r["id"]: r for r in rows}
        self.index = TfidfIndex([(r["id"], r["record"]) for r in rows])

    def shots(self, row, k, compose):
        """Return (messages, retrieved) for the k nearest pool records."""
        hits = self.index.search(row.get("record") or row.get("prompt") or "", k, exclude={row["id"]})
        messages = []
        for doc_id, _ in hits:
            ex = self.rows[doc_id]
            messages.append({"role": "user", "content": compose({"instruction": row.get("instruction", ""), "record": ex["record"]})})
            messages.append({"role": "assistant", "content": json.dumps(ex["label"], ensure_ascii=False)})
        return messages, [{"id": d, "score": s} for d, s in hits]
