### Part D: baseline vs guarded

| Category | n | Baseline success | Guarded success | Catch rate | Over-refusal | Review |
|---|---:|---:|---:|---:|---:|---:|
| toxic | 4 | 0 | 0 | 0% | — | — |
| out_of_scope | 4 | 0 | 0 | 0% | — | — |
| injection | 4 | 1 | 0 | 50% | — | — |
| leakage | 4 | 1 | 0 | 50% | — | — |
| benign | 14 | — | — | — | 0% | 0% |

Guardrail cost on allowed benign rows (n=14): -0.157 s latency, +25.4 tokens per request (baseline 4.876 s, 505.9 tokens).
