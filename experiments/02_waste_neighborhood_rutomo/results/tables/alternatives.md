### Part E: alternatives on EVAL-50

| Run | Settings | n | Valid | Exact [95% CI] | Focus | Type | Dept | κ focus | κ type | Ungrounded dept names | Truncated | Latency mean / p95 (s) | Tokens in / out |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| E_gemma270m_structured | T=1.0, top_p=0.95, max_new=256, sample=False | 50 | 68% | 0% [0, 0] | 44% | 36% | 34% | 0.05 | -0.06 | 31% | 2% | 1.754 / 5.05 | 1351.7 / 105.4 |
| E_nonllm_tfidf_lr | TF-IDF(1-2gram) + LogisticRegression(bal | 50 | 100% | 52% [38, 64] | 60% | 100% | 78% | 0.39 | 1.00 | 0% | 0% | 0.0 / 0.0 | 0 / 0 |
| C1_structured | T=1.0, top_p=0.95, max_new=256, sample=False | 50 | 98% | 56% [42, 70] | 74% | 94% | 74% | 0.59 | 0.90 | 3% | 0% | 6.223 / 11.148 | 670.9 / 43.1 |
| C2_tools | T=1.0, top_p=0.95, max_new=256, sample=False | 50 | 90% | 30% [18, 44] | 74% | 72% | 56% | 0.59 | 0.57 | 25% | 0% | 15.399 / 22.884 | 1786.0 / 105.9 |
| C3_guarded | T=1.0, top_p=0.95, max_new=256, sample=False | 50 | 100% | 66% [52, 78] | 82% | 96% | 82% | 0.70 | 0.93 | 2% | 0% | 5.72 / 8.423 | 658.0 / 39.6 |
| C4_fewshot | T=1.0, top_p=0.95, max_new=256, sample=False | 50 | 98% | 62% [48, 74] | 80% | 96% | 76% | 0.69 | 0.93 | 2% | 0% | 9.713 / 15.752 | 1814.8 / 32.7 |
