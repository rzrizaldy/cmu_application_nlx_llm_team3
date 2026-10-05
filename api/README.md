# Team LLM API

We turn a resident's 311 complaint into a routed ticket, following the brief's pipeline: retrieve domain knowledge, then classify, clarify, and structure.

```python
import sys; sys.path.insert(0, "api")
from team311.model import PhiRunner
from team311.pipeline import Router

router = Router(PhiRunner(), mode="T2_tools", dev_examples=dev_rows)  # dev_rows: experiments/05_team/data/dev.jsonl
out = router.route_complaint("the light by the tennis court in frick park has been out for a week")
out["ticket"]   # Ticket311 fields: domain, category, issue, department, missing_information,
                # clarification_question, confidence, abstain, historical_resolution_range
out["trace"]    # retrieved docs, neighbors, candidate issues, codebook hits, parse status
```

## Pieces

| File | What it does |
|---|---|
| [team311/pipeline.py](team311/pipeline.py) | `Router.route_complaint()`: prompt, generation, JSON parsing, category and issue snapping, codebook grounding |
| [team311/knowledge.py](team311/knowledge.py) | Loads the knowledge base (`corpora/05_all/corpus.jsonl` without intake examples, plus the 127 `knowledge_cards.jsonl`); TF-IDF `Retriever` with document search, labeled DEV neighbors, and Mingchin's issue vote; `resolution_range()` |
| [team311/tools.py](team311/tools.py) | Codebook lookup: issue names that appear verbatim in the complaint |
| [team311/guardrail.py](team311/guardrail.py) | Input rules (injection, prompt leakage, abusive drafting, out of scope) and output redaction of phone numbers and emails |
| [team311/model.py](team311/model.py) | Local Phi-4-mini runner with an optional LoRA adapter |
| [llmbox/src/pydantic_models/ticket311.py](llmbox/src/pydantic_models/ticket311.py) | `Ticket311` schema the experiments validate against |
| [llmbox/](llmbox/) | Rutomo's LLMBox fork with the Assignment 2 changes (M0–M5); models live outside git, see [docs/local_phi_model.md](../docs/local_phi_model.md) |

## Modes

Each mode adds one layer; the team experiments in [experiments/05_team/](../experiments/05_team/) run each on the same 50 EVAL inputs.

| Mode | Prompt | After generation |
|---|---|---|
| `T0_generate` | The 17 allowed categories and the complaint | Nothing |
| `T1_structured_rag` | + 3 retrieved knowledge documents | Category snapped to the allowed list; domain derived from category; resolution range from the cards |
| `T2_tools` | Instead of documents: 3 labeled DEV neighbors, the vote's most likely routing with its required information, other candidates, and codebook name matches | T1, plus the issue is snapped to a candidate; a matched card supplies the codebook category, department, and clarification question; if no card matches, the vote's top issue is used and the ticket is marked `issue_source: vote_fallback` |
| `T3_guarded` | T2 behind the input guardrail | T2, plus output redaction |

T4 is the T0 prompt on a LoRA-adapted Phi-4-mini trained on DEV conversations.
