import json
import time

from omegaconf import OmegaConf
from src.generation import GenerationManager


input_file = "data/pittsburgh311/dataset_splits/evaluation_50_text_inputs.jsonl"
output_file = "outputs/tool_calling_eval2_responses.jsonl"


cfg = OmegaConf.create({
    "model": {
        "name": "phi4_instruct",
        "source": "local",
        "model_id": "microsoft/Phi-4-mini-instruct",
        "local_path": "./models/llms/microsoft/phi-4-mini-instruct",
        "architecture": "phi3",
        "dtype": "bfloat16",
        "device": "auto",
        "trust_remote_code": False,
        "supports_tool_calling": True,
        "supports_structured_output": True,
        "tool_calling_format": "functools_prompt",
        "chat_template_kwargs": {}
    },

    "generation": {
        "max_new_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 64,
        "do_sample": True,
        "repetition_penalty": 1.0
    },

    "tool_calling": {
        "enabled": True,
        "tool_choice": "auto",

        "tools": [
            {
                "name": "lookup_311_taxonomy",
                "description": (
                    "Look up the official Pittsburgh 311 routing labels "
                    "for a service request issue."
                ),

                "parameters": {
                    "type": "object",

                    "properties": {
                        "issue_keyword": {
                            "type": "string",
                            "description": (
                                "Short description of the resident's issue."
                            )
                        }
                    },

                    "required": [
                        "issue_keyword"
                    ]
                }
            }
        ]
    },

    "system_prompt": (
        "You are assisting with Pittsburgh 311 service request routing. "
        "Use the lookup_311_taxonomy tool when it can help identify "
        "the official routing labels."
    )
})


generator = GenerationManager()

print("Loading Phi-4-mini-instruct...")

model, tokenizer, device = generator._load_model_and_tokenizer(cfg)

print("Model loaded on:", device)


records = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip() != "":
            records.append(json.loads(line))


tools = OmegaConf.to_container(
    cfg.tool_calling.tools,
    resolve=True
)


system_content = generator.build_functools_system_prompt(
    cfg.system_prompt,
    tools
)


with open(output_file, "w", encoding="utf-8") as out_file:

    for i, record in enumerate(records, start=1):

        prompt = (
            "Classify this Pittsburgh 311 resident request and provide "
            "the official request type ID, issue, category, and department.\n\n"
            f"Resident request: {record['input']}"
        )

        messages = [
            {
                "role": "system",
                "content": system_content
            },

            {
                "role": "user",
                "content": prompt
            }
        ]

        start_time = time.perf_counter()

        response, tool_calls = generator.run_generic_tool_turn(
            model,
            tokenizer,
            device,
            messages,
            cfg,
            tools,
            cfg.tool_calling.tool_choice
        )

        latency = time.perf_counter() - start_time


        tool_used = False
        tool_result = None

        if len(tool_calls) > 0:
            tool_used = True
            tool_result = tool_calls[0]["result"]


        result = {
            "evaluation": "tool_calling_eval2",

            "doc_id": record["doc_id"],
            "input": record["input"],

            "ground_truth": {
                "request_type_id": record["request_type_id"],
                "issue": record["issue"],
                "category": record["category"],
                "department": record["department"]
            },

            "tool_used": tool_used,
            "tool_calls": tool_calls,
            "tool_result": tool_result,

            "model_response": response,

            "generation_settings": {
                "temperature": 0.2,
                "top_p": 0.8,
                "max_new_tokens": 128
            },

            "latency_seconds": latency
        }


        out_file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

        out_file.flush()


        print(
            f"[{i}/50] {record['doc_id']} "
            f"tool_used={tool_used} "
            f"({latency:.2f}s)"
        )


print()
print("Finished.")
print("Saved responses to:", output_file)