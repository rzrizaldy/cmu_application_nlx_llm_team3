"""
scenario.py -- fixed vocabularies and paths for the Pittsburgh 311 triage API.

This is the single source of truth for the organizational scenario so every
script (Parts B-E) agrees on the categories, the departments, and where the
data splits live. Importing this module does NOT import torch or llmbox, so it
stays cheap to load from anywhere.
"""
from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Organizational scenario (City of Pittsburgh, 311 Operations)
# ---------------------------------------------------------------------------
SCENARIO = "City of Pittsburgh 311 service-request triage and routing"

# The four issue categories from the Assignment 1 corpus (the API classifies
# an incoming request into exactly one of these).
ISSUE_CATEGORIES = [
    "building_maintenance",
    "construction",
    "permits",
    "accessibility",
]

# The departments a request may be routed to. Drawn from the actual
# `responsible_department` values that appear in the corpus; "unknown" is the
# abstain option the policy requires the model to be able to choose rather than
# guessing a route.
DEPARTMENTS = [
    "Permits, Licenses and Inspections",
    "DOMI - Permits",
    "DPW - Street Maintenance",
    "Finance",
    "unknown",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CORPUS_PATH = HERE / "data" / "corpus.jsonl"      # copied in by make_splits.py
DEV_PATH = HERE / "data" / "dev.jsonl"            # 50 records, Part B (tuning)
EVAL_PATH = HERE / "data" / "eval.jsonl"          # 50 records, Part C/D (held out)
OUT_DIR = HERE / "out"                            # all run artifacts land here

# Which corpus field is the model's INPUT. 
INPUT_FIELD = "raw_text"


def ensure_dirs() -> None:
    (HERE / "data").mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
