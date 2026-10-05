# Running the task through the LLMBox CLI (native-mode proof)

These commands run your triage task through the professor's LLMBox directly
(`startllm.py`), proving the custom features work in LLMBox's own modes. The
CLI reloads the model on every call (~60s each from Assignment 1), so these are
for a few proof records; the 50-input graded batches use the wrapper in
`triage_api.py`, which imports LLMBox's own `GenerationManager`
(`_load_model_and_tokenizer` + `_generate_once`) and runs the identical code
path without reloading.

## Setup (one time)
```
# from the unzipped llmbox-main/ folder, with your Phi path:
set PHI=C:\Users\mahik\Downloads\phi4-mini-instruct\phi4-mini-instruct   # Windows
# copy the two new files into the LLMBox tree (NEW files, no edits to her code):
copy ..\assignment2\llmbox_integration\pydantic_models\triage.py src\pydantic_models\triage.py
copy ..\assignment2\llmbox_integration\conf\tool_calling\route_request.yaml conf\tool_calling\route_request.yaml
python -c "from src.pydantic_models.triage import export; print(export('conf/triage_schema.json'))"
```

## mode=generate (Part B baseline, one record)
```
python startllm.py model=phi4_instruct mode=generate model.source=local model.local_path=%PHI% ^
  generation.temperature=0.0 generation.top_p=1.0 generation.max_new_tokens=128 ^
  system_prompt="You are a triage assistant for the City of Pittsburgh 311 service center. Read one 311 service request and classify it." ^
  prompt="311 request: A resident submitted a 311 service request about \"Building Without a Permit\" in the South Side Slopes neighborhood of Pittsburgh. Return JSON with keys issue_category (one of building_maintenance, construction, permits, accessibility) and responsible_department. Return only the JSON."
```

## mode=structured_output (Part C Eval 1, one record)
```
python startllm.py model=phi4_instruct mode=structured_output structured_output.enabled=true ^
  structured_output.schema_path=conf/triage_schema.json model.source=local model.local_path=%PHI% ^
  generation.temperature=0.0 ^
  prompt="311 request: Broken sidewalk and curb ramp missing on Wood St in the Central Business District."
```

## mode=tool_calling (Part C Eval 2, one record)
```
# route_request follows LLMBox's TOOL_REGISTRY pattern (see assignment2/triage_tools.py,
# modeled on her src/tools.py). To exercise it via the CLI, register route_request in
# src/tools.py the same way get_current_weather is registered, then:
python startllm.py model=phi4_instruct mode=tool_calling tool_calling=route_request ^
  model.source=local model.local_path=%PHI% ^
  prompt="Classify and route: unpermitted electrical work in Lawrenceville."
```

Save the terminal output of each as a screenshot or text file; those are your
"used the LLMBox modes" evidence for the memo appendix.
