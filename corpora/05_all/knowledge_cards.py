#!/usr/bin/env python3
"""One knowledge card per issue, with the knowledge fields the brief lists.

Built from operational_evidence.jsonl (codebook labels, definition, resolution
times). Confusable issues are the three most similar issue names (character
n-gram TF-IDF on issue name, aliases, and definition) that route to a different category or
department. Required information and the clarification question are
team-authored per category in CATEGORY_GUIDANCE, since no member corpus
provides them per issue.
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from category_groups import CATEGORY_GROUPS

ALL_DIR = Path(__file__).resolve().parent
OPS = ALL_DIR / "operational_evidence.jsonl"
OUT = ALL_DIR / "knowledge_cards.jsonl"

CATEGORY_GUIDANCE = {
    "Road/Street Issues": (
        ["street and nearest cross street", "size of the problem", "whether it blocks a travel lane"],
        "Which street and nearest cross street is it on, and is it blocking a travel lane?",
    ),
    "Traffic and Street Sign Issues": (
        ["intersection", "sign or signal type", "missing, damaged, or not working"],
        "At which intersection is the sign or signal, and is it missing, damaged, or not working?",
    ),
    "Street Light": (
        ["nearest address or pole number", "out, flickering, or on during the day"],
        "Where is the light (nearest address or pole number), and is it out, flickering, or on during the day?",
    ),
    "Pedestrian/Bicycle Concerns": (
        ["exact location", "sidewalk, curb ramp, steps, or bike lane", "whether it blocks access"],
        "Where exactly is it, and is it a sidewalk, curb ramp, city steps, or bike lane?",
    ),
    "Parking": (
        ["block or nearest address", "how long the vehicle has been there", "whether it blocks a driveway, hydrant, or lane"],
        "Where is the vehicle, how long has it been there, and is it blocking a driveway, hydrant, or lane?",
    ),
    "Garbage and Litter Issues": (
        ["block or nearest address", "trash, recycling, or bulk item", "scheduled collection day"],
        "Was it trash, recycling, or a bulk item, and what is your scheduled collection day?",
    ),
    "Neighborhood Issues": (
        ["property location", "type of condition", "how long it has been there"],
        "What is the property location, and how long has the condition been there?",
    ),
    "Graffiti Issues": (
        ["location", "public property or private building", "offensive content"],
        "Where is the graffiti, and is it on public property or a private building?",
    ),
    "Weeds/Debris": (
        ["location", "city lot, private lot, or right of way", "size of the area"],
        "Is the overgrowth or debris on a city lot, a private lot, or the public right of way?",
    ),
    "Building Maintenance": (
        ["property location", "condition observed", "occupied or vacant", "immediate safety hazard"],
        "What is the property location, and is there an immediate safety hazard?",
    ),
    "Construction Issues": (
        ["site location", "type of work", "whether work is active now", "whether a permit is posted"],
        "What kind of work is happening, and is a permit posted at the site?",
    ),
    "Permits": (
        ["property location", "type of work or permit"],
        "Which property and what type of work or permit is this about?",
    ),
    "Accessibility": (
        ["location", "type of barrier", "whether it blocks wheelchair or mobility-device access"],
        "Where is the barrier, and does it block wheelchair or mobility-device access?",
    ),
    "Parks Issues": (
        ["park name", "feature affected (court, playground, shelter, lights)", "safety hazard"],
        "Which park and which feature (court, playground, shelter, lights) is affected?",
    ),
    "Tree Issues": (
        ["location", "city tree or private tree", "whether it blocks the street or sidewalk", "fallen or standing"],
        "Is the tree in the public right of way or on private property, and is it blocking the street or sidewalk?",
    ),
    "Animal Issues": (
        ["location", "type of animal", "alive, injured, or dead", "immediate danger"],
        "What kind of animal is it, and is it alive, injured, or dead?",
    ),
    "City Facilities and Infrastructure": (
        ["facility or steps location", "problem observed", "safety hazard"],
        "Which city facility or steps are affected, and what is the problem?",
    ),
}
assert set(CATEGORY_GUIDANCE) == {c for cats in CATEGORY_GROUPS.values() for c in cats}


def main() -> int:
    ops = [json.loads(line) for line in OPS.read_text().splitlines() if line.strip()]
    stats = [r["table_json"] for r in ops]
    defs = [r["raw_text"].split(" Definition: ", 1)[1] if " Definition: " in r["raw_text"] else "" for r in ops]
    texts = [" ".join([s["issue"], *s["aliases"], d]) for s, d in zip(stats, defs)]
    sim = cosine_similarity(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(texts))

    cards = []
    for i, (rec, s, definition) in enumerate(zip(ops, stats, defs)):
        confusable = [
            {"issue": stats[j]["issue"], "category": stats[j]["category"], "department": stats[j]["department"]}
            for j in sim[i].argsort()[::-1]
            if j != i and sim[i][j] > 0.15
            and (stats[j]["category"], stats[j]["department"]) != (s["category"], s["department"])
        ][:3]
        required, question = CATEGORY_GUIDANCE[s["category"]]
        card = {
            "request_type_id": s["request_type_id"],
            "issue": s["issue"],
            "aliases": s["aliases"],
            "category": s["category"],
            "department": s["department"],
            "definition": definition or None,
            "includes": [s["issue"], *s["aliases"]],
            "confusable_issues": confusable,
            "required_information": required,
            "clarification_question": question,
            **{k: s[k] for k in ("request_volume", "closed_requests", "median_days", "p75_days", "p90_days") if k in s},
        }
        text = [f"Issue: {s['issue']} (request type {s['request_type_id']}). Category: {s['category']}. Department: {s['department']}."]
        if s["aliases"]:
            text.append("Also filed as: " + "; ".join(s["aliases"]) + ".")
        if definition:
            text.append(f"Definition: {definition}")
        if confusable:
            text.append("Not to be confused with: " + "; ".join(
                f"{c['issue']} ({c['category']}, {c['department']})" for c in confusable) + ".")
        text.append("Required information: " + "; ".join(required) + ".")
        text.append(f"Clarification question: {question}")
        if "median_days" in s:
            text.append(f"Historical resolution: median {s['median_days']} days, "
                        f"75th percentile {s['p75_days']}, 90th percentile {s['p90_days']} days.")
        cards.append({
            "doc_id": f"card_{s['request_type_id']}",
            "source_url": rec["source_url"],
            "retrieved_at": rec["retrieved_at"],
            "modality": "mixed",
            "raw_text": " ".join(text),
            "table_json": card,
            "license_note": rec["license_note"],
            "metadata": {
                **rec["metadata"],
                "document_type": "knowledge_card",
                "derived_from": rec["doc_id"],
                "team_authored_fields": ["required_information", "clarification_question"],
            },
        })

    OUT.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cards))
    print(f"{len(cards)} knowledge cards -> {OUT.name}; "
          f"{sum(1 for c in cards if c['table_json']['definition'])} with a codebook definition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
