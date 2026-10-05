### Part E: cost model (per record)

Manual: 18.0 s/record; review: 12.0 s/output.

| Run | Usable | Usable basis | Tokens/req | Compute s | Human s with API | Manual $ | API $ | Saving $ | Break-even records |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| C1_structured | 56% | exact_match_proxy | 714.0 | 6.223 | 19.9 | 0.21 | 0.2324 | -0.0224 | None |
| C3_guarded | 75% | timed_human_review | 697.5 | 5.563 | 16.5 | 0.21 | 0.1925 | 0.0175 | 144025 |
| C4_fewshot | 62% | exact_match_proxy | 1847.5 | 9.713 | 18.8 | 0.21 | 0.2198 | -0.0098 | None |
| C2_tools | 30% | exact_match_proxy | 1891.9 | 15.399 | 24.6 | 0.21 | 0.287 | -0.077 | None |
