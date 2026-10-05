### Part B: baseline generate on DEV-50

| Run | Settings | n | Valid | Exact [95% CI] | Focus | Type | Dept | κ focus | κ type | Ungrounded dept names | Truncated | Latency mean / p95 (s) | Tokens in / out |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| B1_generate_defaults | T=1.0, top_p=0.95, max_new=512, sample=True | 50 | 54% | 18% [8, 30] | 38% | 38% | 38% | 0.29 | 0.13 | 14% | 0% | 5.554 / 12.668 | 351.0 / 43.6 |
| B3_generate_tuned | T=0.2, top_p=0.9, max_new=256, sample=True | 50 | 74% | 26% [14, 38] | 54% | 52% | 58% | 0.44 | 0.23 | 4% | 0% | 4.631 / 6.264 | 351.0 / 39.4 |
