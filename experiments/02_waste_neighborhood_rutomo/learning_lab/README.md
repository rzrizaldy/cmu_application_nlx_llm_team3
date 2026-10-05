# A1 → A2 local learning lab

From the Assignment 2 directory on the Mac:

```sh
.venv/bin/python a2/learning_lab/server.py
```

Open `http://127.0.0.1:8762/`. The server binds to loopback and loads local model weights on the first request. Phi is the default; Gemma is available for a quicker small-model comparison. Stop it with Ctrl-C. The A1 link serves the existing Assignment 1 learning lab from this course repository.

The live form uses the vendored LLMBox generation, structured-output, tool, and guardrail paths. Tools require Phi and are limited to read-only lookups. Requests are serialized because the model is shared in memory. The lab reads saved run counts and scored metrics when available; it does not import labels, read the evaluation pool, or write to graded runs. It does not save practice text, model responses, or interaction logs locally or on SSH. The hypothetical reviewer scenario exists only in the open browser tab; it is not an actual interaction record.

The cost sandbox uses the same per-record formula as `a2/scripts/cost.py` but saves nothing. Its inputs are hypothetical until the student's labeling and review timings and sourced assumptions have been entered. Run scores use `metrics.json` against the student reference labels when present.

Run from the Assignment 2 directory so the configured model paths resolve. No hosted model is used. The lab intentionally supplies questions and code pointers for self-study, not assignment report prose or reference labels.
