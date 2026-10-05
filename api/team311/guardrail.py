"""Rule-based intake guardrail (team scope).

Input side blocks injection, prompt leakage, toxic drafting, and out-of-scope
actions before the model runs. Output side redacts phone numbers and emails.
"""
from __future__ import annotations

import re

INJECTION = re.compile(
    r"ignore (all )?(previous|prior|the) (instructions|triage|311)|disregard the|"
    r"new instructions|forget your rules|unfiltered model|\[\[system\]\]|note to system|"
    r"zzcanary|drop table",
    re.I,
)
LEAKAGE = re.compile(
    r"system prompt|hidden (system )?instructions|your (full )?instructions|repeat everything above|"
    r"rules you were given|restricted fields|masked spans|internal (reference )?marker|internal routing codes",
    re.I,
)
TOXIC = re.compile(
    r"\b(insult\w*|demeaning|abusive|hostile|mock\w*|worthless|deadbeat|scum|slum|filthy|"
    r"idiot\w*|stupid|lazy|deserves to rot)\b",
    re.I,
)
OUT_OF_SCOPE = re.compile(
    r"home address|phone number|give me (their|the) name|legal advice|press release|"
    r"predict which neighborhoods|issue a (real )?citation|approve a refund|charge it to|"
    r"committed a crime|weather|bitcoin|recipe|homework",
    re.I,
)

PHONE = re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")


def check_input(text: str) -> dict | None:
    if INJECTION.search(text):
        return {"status": "refused", "reason": "Prompt injection pattern blocked."}
    if LEAKAGE.search(text):
        return {"status": "refused", "reason": "Request for system instructions or restricted data."}
    if TOXIC.search(text):
        return {"status": "refused", "reason": "Request asks for abusive content."}
    if OUT_OF_SCOPE.search(text):
        return {"status": "out_of_scope", "reason": "Request is outside Pittsburgh 311 intake scope."}
    return None


def redact_output(text: str) -> str:
    return EMAIL.sub("[EMAIL]", PHONE.sub("[PHONE]", text))
