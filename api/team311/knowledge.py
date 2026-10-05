"""Team knowledge base and retrieval.

The knowledge base is what the brief describes: the four member corpora
concatenated, plus one knowledge card per issue (corpora/05_all/knowledge_cards.jsonl).
Labeled intake examples from the member corpora are not knowledge documents;
DEV intake examples can be added separately as labeled neighbors for issue voting.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "corpora" / "05_all" / "corpus.jsonl"
CARDS = REPO / "corpora" / "05_all" / "knowledge_cards.jsonl"

INTAKE_MARKERS = {
    "afaq": lambda m: m.get("document_type") == "complaint",
    "mahika": lambda m: m.get("document_type") == "service_request",
    "mingchin": lambda m: bool(m.get("example_style")),
    "rutomo": lambda m: m.get("source_kind") == "issue_taxonomy",
}


def _content(record: dict) -> str:
    text = record.get("raw_text") or ""
    table = record.get("table_json")
    if table is not None and record["metadata"].get("member") != "team":
        text += "\n" + json.dumps(table, ensure_ascii=False)
    return text.strip()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def load_knowledge(corpus_path: Path = CORPUS, cards_path: Path = CARDS) -> list[dict]:
    out = []
    for r in _jsonl(corpus_path) + _jsonl(cards_path):
        m = r["metadata"]
        if INTAKE_MARKERS.get(m["member"], lambda _: False)(m):
            continue
        out.append({
            "doc_id": r["doc_id"],
            "subtopic_key": m["subtopic_key"],
            "text": _content(r),
            "kind": m.get("document_type") or m.get("source_kind") or m.get("record_type") or "record",
            "card": r["table_json"] if m["member"] == "team" else None,
        })
    return out


def cards(index: list[dict]) -> dict[str, dict]:
    """Issue name (lowercase, without a '(DO NOT USE)' suffix) to knowledge card."""
    return {issue_key(d["card"]["issue"]): d["card"] for d in index if d["card"]}


def issue_key(name: str | None) -> str:
    name = (name or "").strip().lower()
    return name[: -len("(do not use)")].strip() if name.endswith("(do not use)") else name


def find_card(index: list[dict], issue: str | None) -> dict | None:
    by_name = cards(index)
    key = issue_key(issue)
    if key in by_name:
        return by_name[key]
    for card in by_name.values():
        if key and key in {issue_key(a) for a in card.get("aliases", [])}:
            return card
    return None


class Retriever:
    """TF-IDF over knowledge documents plus labeled DEV neighbors.

    search() returns knowledge documents for the prompt. vote_issues() follows
    Mingchin's majority-vote router: the nearest labeled DEV examples and cards
    vote for their issue, weighted by cosine similarity.
    """

    def __init__(self, index: list[dict], neighbors: list[dict] | None = None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.index = index
        self.neighbors = neighbors or []
        self.card_by_issue = cards(index)
        texts = [d["text"] for d in index] + [n["input"] for n in self.neighbors]
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        matrix = self.vectorizer.fit_transform(texts)
        self.doc_matrix = matrix[: len(index)]
        self.neighbor_matrix = matrix[len(index):]

    def _scores(self, query: str, matrix):
        if matrix.shape[0] == 0:
            return []
        return (matrix @ self.vectorizer.transform([query]).T).toarray().ravel()

    def search(self, query: str, k: int = 3) -> list[dict]:
        scores = self._scores(query, self.doc_matrix)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.index[i] for i in order if scores[i] > 0]

    def nearest(self, query: str, k: int = 3) -> list[dict]:
        scores = self._scores(query, self.neighbor_matrix)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.neighbors[i] for i in order if scores[i] > 0]

    def vote_issues(self, query: str, k_neighbors: int = 25, top: int = 8) -> list[dict]:
        votes: dict[str, float] = defaultdict(float)
        n_scores = self._scores(query, self.neighbor_matrix)
        for i in sorted(range(len(n_scores)), key=lambda i: -n_scores[i])[:k_neighbors]:
            if n_scores[i] > 0:
                votes[issue_key(self.neighbors[i]["gold"]["issue"])] += float(n_scores[i])
        d_scores = self._scores(query, self.doc_matrix)
        for i in sorted(range(len(d_scores)), key=lambda i: -d_scores[i])[:k_neighbors]:
            card = self.index[i]["card"]
            if card and d_scores[i] > 0:
                votes[issue_key(card["issue"])] += float(d_scores[i])
        ranked = sorted(((v, k) for k, v in votes.items() if k in self.card_by_issue), reverse=True)[:top]
        return [{**self.card_by_issue[k], "vote": round(v, 3)} for v, k in ranked]


def _fmt_days(d: float) -> str:
    return f"{d * 24:.0f} hours" if d < 1 else f"{d:.1f} days"


def resolution_range(index: list[dict], issue: str | None = None, category: str | None = None) -> str | None:
    """Historical closure time from the WPRDC operational evidence on the cards.

    Exact issue match first; otherwise the volume-weighted median across the
    category's issues, with the 75th-90th percentile span of its busiest issue.
    """
    card = find_card(index, issue)
    if card and "median_days" in card:
        return (f"{card['issue']}: median {_fmt_days(card['median_days'])}, "
                f"75th-90th percentile {_fmt_days(card['p75_days'])}-{_fmt_days(card['p90_days'])} "
                f"({card['closed_requests']} closed requests)")
    if category:
        rows = sorted((d["card"] for d in index if d["card"] and d["card"]["category"] == category
                       and "median_days" in d["card"]), key=lambda c: -c["closed_requests"])
        if rows:
            total = sum(c["closed_requests"] for c in rows)
            median = sum(c["median_days"] * c["closed_requests"] for c in rows) / max(total, 1)
            top = rows[0]
            return (f"{category}: median about {_fmt_days(median)} across {len(rows)} issues "
                    f"({total} closed requests); busiest issue {top['issue']} "
                    f"75th-90th percentile {_fmt_days(top['p75_days'])}-{_fmt_days(top['p90_days'])}")
    return None
