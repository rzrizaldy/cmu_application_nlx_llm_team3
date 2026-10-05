### Ablations (DEV-50)

| Run | Settings | n | Valid | Exact [95% CI] | Focus | Type | Dept | κ focus | κ type | Ungrounded dept names | Truncated | Latency mean / p95 (s) | Tokens in / out |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| C1abl_dev_dept_optional | T=1.0, top_p=0.95, max_new=256, sample=False | 50 | 100% | 46% [32, 60] | 84% | 88% | 70% | 0.75 | 0.79 | 0% | 0% | 5.9 / 8.197 | 618.0 / 39.4 |
| C1abl_dev_dept_required | T=1.0, top_p=0.95, max_new=256, sample=False | 50 | 100% | 28% [16, 40] | 82% | 88% | 52% | 0.72 | 0.79 | 6% | 0% | 6.315 / 10.805 | 715.8 / 44.1 |
| C2abl_dev_tools_auto | T=1.0, top_p=0.95, max_new=256, sample=False | 50 | 100% | 56% [42, 70] | 82% | 94% | 76% | 0.73 | 0.89 | 2% | 0% | 6.141 / 8.467 | 910.2 / 38.8 |
| C2abl_dev_tools_required | T=1.0, top_p=0.95, max_new=256, sample=False | 50 | 70% | 26% [14, 38] | 58% | 52% | 52% | 0.46 | 0.35 | 18% | 0% | 16.301 / 25.591 | 1768 / 119.5 |
