"""
triage_tools.py -- Part C, Custom Evaluation 2: a tool the model can call to
route a 311 request, plus the LLMBox tool-config block.

This mirrors the professor's src/tools.py convention (a TOOL_REGISTRY populated
by a @register_tool decorator) but lives in your own file so her repo is not
modified. The tool, route_request, takes an issue_category and returns the
responsible department from a deterministic routing table. The organizational
point: the LLM decides the category (the hard, learned part) and the TOOL owns
the routing policy (the part that must be correct and auditable, so it should
NOT be left to the model).

In mode=tool_calling, LLMBox shows the model the tool definition
(TOOL_CONFIG['tools']) and, when the model emits functools[...] asking for
route_request, GenerationManager looks the name up in a registry and runs it.
For the batch Part C run we also expose route_request directly so run_partC.py
can execute the tool over 50 inputs without the CLI.
"""
from __future__ import annotations
from typing import Optional

import scenario

TOOL_REGISTRY: dict = {}


def register_tool(name):
    def deco(fn):
        TOOL_REGISTRY[name] = fn
        return fn
    return deco


# Deterministic routing table: category -> default department. This is the
# organization's routing POLICY, encoded in code rather than left to the model.
ROUTING_TABLE = {
    "permits": "Permits, Licenses and Inspections",
    "building_maintenance": "Permits, Licenses and Inspections",
    "construction": "Permits, Licenses and Inspections",
    "accessibility": "DOMI - Permits",
}


@register_tool("route_request")
def route_request(issue_category: str, neighborhood: Optional[str] = None) -> dict:
    """Route a classified 311 request to the responsible department.

    Args:
        issue_category: one of building_maintenance, construction, permits, accessibility.
        neighborhood: optional Pittsburgh neighborhood (unused by the base table,
            present so the tool signature can grow without breaking callers).
    Returns:
        {"issue_category", "responsible_department", "routed"} where routed is
        False and department is "unknown" when the category is unrecognized --
        the tool abstains rather than guessing.
    """
    cat = (issue_category or "").strip().lower().replace(" ", "_").replace("-", "_")
    dept = ROUTING_TABLE.get(cat)
    if dept is None:
        return {"issue_category": cat, "responsible_department": "unknown", "routed": False}
    return {"issue_category": cat, "responsible_department": dept, "routed": True}


# LLMBox tool-calling config block (same shape as conf/tool_calling/weather.yaml).
TOOL_CONFIG = {
    "enabled": True,
    "tool_choice": "auto",
    "tools": [
        {
            "name": "route_request",
            "description": "Route a classified Pittsburgh 311 request to the "
                           "responsible city department.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_category": {
                        "type": "string",
                        "enum": scenario.ISSUE_CATEGORIES,
                    },
                    "neighborhood": {"type": "string"},
                },
                "required": ["issue_category"],
            },
        }
    ],
}


def write_tool_yaml(path=None):
    """Optional: emit a conf/tool_calling/route_request.yaml so the tool can
    also be driven from the LLMBox CLI. Writes into YOUR folder, not hers."""
    import yaml
    path = path or (scenario.HERE / "data" / "route_request.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(TOOL_CONFIG, f, sort_keys=False)
    return path


if __name__ == "__main__":
    print(route_request("permits"))
    print(route_request("accessibility"))
    print(route_request("garbage"))  # abstains
