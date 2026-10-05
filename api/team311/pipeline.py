"""Team 311 intake API: resident complaint -> routed Ticket311.

    from team311.model import PhiRunner
    from team311.pipeline import Router
    router = Router(PhiRunner(), mode="T2_tools")
    router.route_complaint("the light in frick park by the tennis court is out")

Modes add one layer each, matching the team experiments:
  T0_generate        prompt with the 17 allowed categories only
  T1_structured_rag  + 3 retrieved knowledge documents; category snapped to the allowed list
  T2_tools           + candidate issues from the TF-IDF vote and the codebook lookup tool;
                     a predicted issue that matches a card takes its codebook category,
                     department, clarification question, and resolution range
  T3_guarded         T2 behind the input guardrail and output redaction
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "corpora" / "05_all"))
from category_groups import CATEGORY_GROUPS  # noqa: E402

from .guardrail import check_input, redact_output
from .knowledge import Retriever, find_card, issue_key, load_knowledge, resolution_range
from .tools import lookup_codebook

MODES = ("T0_generate", "T1_structured_rag", "T2_tools", "T3_guarded")
ALL_CATEGORIES = [c for cats in CATEGORY_GROUPS.values() for c in cats]
CATEGORY_TO_DOMAIN = {c: sk for sk, cats in CATEGORY_GROUPS.items() for c in cats}
TICKET_KEYS = ("domain", "category", "issue", "department", "missing_information",
               "clarification_question", "confidence", "abstain", "historical_resolution_range")

SYSTEM = """You turn a Pittsburgh resident's 311 complaint into one routed ticket. Output a single JSON object only, no prose.
Keys: domain, category, issue, department, missing_information, clarification_question, confidence, abstain, historical_resolution_range.
- domain and category must come from the allowed list.
- missing_information lists details a crew would need that the complaint does not give; clarification_question asks for the most important one, or is null if nothing is missing.
- Set abstain true with low confidence when the text is not a city service request.
Format: {"domain":"<domain>","category":"<category>","issue":"<311 issue name>","department":"<city department>","missing_information":[],"clarification_question":null,"confidence":0.8,"abstain":false,"historical_resolution_range":null}"""

Generate = Callable[[str, str], tuple]


def allowed_block() -> str:
    return "Allowed domains and categories:\n" + "\n".join(
        f"- {domain}: " + "; ".join(cats) for domain, cats in CATEGORY_GROUPS.items()
    )


def t0_prompt(text: str) -> str:
    """The T0 user prompt; also the prompt the T4 LoRA adapter is trained and evaluated on."""
    return allowed_block() + "\n\nComplaint:\n" + text[:2000]


def parse_json(text: str) -> dict | None:
    start = (text or "").find("{")
    while start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


FIELD = re.compile(r'"(\w+)"\s*:\s*("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)')


def salvage_json(text: str) -> dict | None:
    """Recover complete key/value pairs from a truncated JSON object."""
    if "{" not in (text or ""):
        return None
    out = {}
    for key, value in FIELD.findall(text[text.find("{"):]):
        try:
            out.setdefault(key, json.loads(value))
        except json.JSONDecodeError:
            continue
    return out or None


def snap_category(pred: dict) -> dict:
    """Map a near-miss category onto the allowed list and derive the domain from it."""
    cat = pred.get("category")
    if isinstance(cat, str) and cat not in ALL_CATEGORIES:
        lowered = {a.lower(): a for a in ALL_CATEGORIES}
        key = cat.replace("_", " ").strip().lower()
        match = lowered.get(key) or next(iter(difflib.get_close_matches(key, list(lowered), n=1, cutoff=0.6)), None)
        if match:
            pred = {**pred, "category": lowered.get(match, match), "category_raw": cat}
    if pred.get("category") in CATEGORY_TO_DOMAIN:
        pred = {**pred, "domain": CATEGORY_TO_DOMAIN[pred["category"]]}
    return pred


def snap_issue(pred: dict, candidates: list[str]) -> dict:
    """Map a paraphrased issue onto the offered candidates: containment first, then close string match."""
    issue = pred.get("issue")
    if not isinstance(issue, str) or not candidates or issue in candidates:
        return pred
    low = issue.lower()
    contained = [c for c in candidates if issue_key(c) and (issue_key(c) in low or low in issue_key(c))]
    match = max(contained, key=len) if contained else next(
        iter(difflib.get_close_matches(issue, candidates, n=1, cutoff=0.6)), None)
    return {**pred, "issue": match, "issue_raw": issue} if match else pred


class Router:
    def __init__(self, generate: Generate, mode: str = "T2_tools", dev_examples: list[dict] | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.generate = generate
        self.mode = mode
        self.index = load_knowledge()
        self.retriever = Retriever(self.index, dev_examples)

    def build_prompt(self, text: str) -> tuple[str, dict]:
        trace: dict = {}
        if self.mode == "T0_generate":
            return t0_prompt(text), trace
        parts = [allowed_block()]
        if self.mode == "T1_structured_rag":
            docs = self.retriever.search(text, k=3)
            trace["retrieved"] = [d["doc_id"] for d in docs]
            if docs:
                parts.append("Retrieved knowledge:\n" + "\n---\n".join(d["text"][:400] for d in docs))
        else:
            parts += self._tool_context(text, trace)
        parts.append("Complaint:\n" + text[:2000])
        return "\n\n".join(parts), trace

    def _tool_context(self, text: str, trace: dict) -> list[str]:
        """T2/T3: labeled neighbors, the vote's routing, and the codebook lookup, kept short for a 3.8B model."""
        parts = []
        neighbors = self.retriever.nearest(text, k=3)
        cands = self.retriever.vote_issues(text)
        exact = [h for h in lookup_codebook(text, ALL_CATEGORIES) if h["issue"] not in {c["issue"] for c in cands}]
        trace["neighbors"] = [n["doc_id"] for n in neighbors]
        trace["candidates"] = [c["issue"] for c in cands]
        trace["codebook_hits"] = [h["issue"] for h in exact]
        route = lambda x: f"{x['issue']} | {x['category']} | {x['department']}"  # noqa: E731
        if neighbors:
            parts.append("Similar past requests (issue | category | department):\n" + "\n".join(
                f"{i}. \"{n['input'][:160]}\" -> {route(n['gold'])}" for i, n in enumerate(neighbors, 1)))
        if cands:
            top = cands[0]
            parts.append(f"Most likely routing by vote: {route(top)}\n"
                         f"Required information for {top['issue']}: {'; '.join(top['required_information'])}")
            others = [route(c) for c in cands[1:]] + [route(h) + " (exact name match)" for h in exact]
            if others:
                parts.append("Other candidates:\n" + "\n".join(f"- {o}" for o in others))
        parts.append("Use the most likely routing unless the complaint clearly fits another candidate better. "
                     "Copy the chosen issue, category, and department exactly.")
        return parts

    def finish(self, pred: dict, candidates: list[str] | None = None) -> dict:
        if self.mode == "T0_generate":
            return pred
        pred = snap_category(pred)
        if self.mode in ("T2_tools", "T3_guarded"):
            pred = snap_issue(pred, candidates or [])
            card = find_card(self.index, pred.get("issue"))
            if not card and candidates:
                card = find_card(self.index, candidates[0])
                pred = {**pred, "issue_raw": pred.get("issue_raw", pred.get("issue")), "issue_source": "vote_fallback"}
            if card:
                pred = {**pred, "issue": card["issue"], "category": card["category"],
                        "department": card["department"], "domain": CATEGORY_TO_DOMAIN[card["category"]],
                        "grounded": True}
                if pred.get("missing_information") and not pred.get("clarification_question"):
                    pred["clarification_question"] = card["clarification_question"]
        if pred.get("category") and not pred.get("historical_resolution_range"):
            pred["historical_resolution_range"] = resolution_range(
                self.index, issue=pred.get("issue"), category=pred.get("category"))
        return pred

    def route_complaint(self, text: str) -> dict:
        """Returns {"ticket": dict, "raw": str, "trace": dict, latency and token counts}."""
        if self.mode == "T3_guarded":
            blocked = check_input(text)
            if blocked:
                return {"ticket": {"abstain": True, "confidence": 0.0, **blocked}, "raw": "",
                        "trace": {"guardrail": blocked}, "latency_s": 0.0, "prompt_tokens": 0, "output_tokens": 0}
        user, trace = self.build_prompt(text)
        raw, latency, p_tok, o_tok = self.generate(SYSTEM, user)
        if self.mode == "T3_guarded":
            raw = redact_output(raw)
        pred = parse_json(raw)
        trace["parsed"] = "json" if pred is not None else None
        if pred is None and self.mode != "T0_generate":
            pred = salvage_json(raw)
            trace["parsed"] = "salvaged" if pred is not None else None
        offered = trace.get("candidates", []) + trace.get("codebook_hits", [])
        ticket = self.finish(pred, offered) if pred is not None else {}
        return {"ticket": ticket, "raw": raw[:2000], "trace": trace,
                "latency_s": round(latency, 3), "prompt_tokens": p_tok, "output_tokens": o_tok}
