# Appendix — AI Code-Assistance Log
 This file is the summary index of what was requested and produced.


## Summary of code-assistance requests

| # | What was requested | File(s) produced |
|---|---|---|
| 1 | Read LLMBox, map its modes, design an API wrapper that reuses its code without editing it | triage_api.py |
| 2 | Scenario constants + reproducible dev/eval split from the A1 corpus | scenario.py, make_splits.py |
| 3 | Shared task prompt, tolerant JSON parser, scoring | task.py |
| 4 | Part B: two baseline generate runs at different hyperparameters + metrics | run_partB.py |
| 5 | Part C Eval 1: Pydantic TriageResult model + JSON schema | triage_models.py, llmbox_integration/pydantic_models/triage.py |
| 6 | Part C Eval 2: route_request tool (LLMBox TOOL_REGISTRY pattern) | triage_tools.py, llmbox_integration/conf/tool_calling/route_request.yaml |
| 7 | Part C Eval 3: category-definition recovery prompt | task.py (SYSTEM_V2), run_partC.py |
| 8 | Part C Eval 4 / Part D: guardrail (input+output checks, refusal, logging) | guardrail.py |
| 9 | Part D: scenario probe set (toxic/out-of-scope/injection+canary/leakage/benign) | make_probes.py |
| 10 | Part D: baseline-vs-guardrail measurement, catch/over-refusal/cost | run_partD.py |
| 11 | Part D: human-adjudication scorer error | score_adjudication.py |
| 12 | Part E: cost/benefit + break-even | compute_costbenefit.py |


