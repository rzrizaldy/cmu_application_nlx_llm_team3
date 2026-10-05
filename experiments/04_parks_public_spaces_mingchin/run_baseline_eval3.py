import json
import time
from pathlib import Path

from omegaconf import OmegaConf
from src.generation import GenerationManager


INPUT_PATH = Path(
    "data/pittsburgh311/dataset_splits/baseline_50_text_inputs.jsonl"
)

OUTPUT_PATH = Path(
    "outputs/baseline_eval3_responses.jsonl"
)


def main():
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
            "chat_template_kwargs": {},
        },

        "generation": {
            "max_new_tokens": 128,
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 64,
            "do_sample": True,
            "repetition_penalty": 1.0,
        },

        "system_prompt": (
            "You are assisting with Pittsburgh 311 service request routing. "
            "Given a resident's request, identify the most appropriate "
            "request type, issue, category, and department."
        ),
    })

    generator = GenerationManager()

    print("Loading Phi-4-mini-instruct...")

    model, tokenizer, device = generator._load_model_and_tokenizer(cfg)

    print("Model loaded on:", device)

    records = []

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as out_file:

        for i, record in enumerate(records, start=1):

            prompt = (
                f"Resident request:\n{record['input']}\n\n"
                "Return your answer with these four fields:\n"
                "request_type_id:\n"
                "issue:\n"
                "category:\n"
                "department:"
            )

            messages = [
                {
                    "role": "system",
                    "content": cfg.system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            start_time = time.perf_counter()

            response = generator._generate_once(
                model,
                tokenizer,
                device,
                messages,
                cfg,
            )

            latency_seconds = time.perf_counter() - start_time

            result = {
                "evaluation": "baseline_eval3",

                "doc_id": record["doc_id"],
                "input": record["input"],

                "ground_truth": {
                    "request_type_id": record["request_type_id"],
                    "issue": record["issue"],
                    "category": record["category"],
                    "department": record["department"],
                },

                "model_response": response,

                "generation_settings": {
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "max_new_tokens": 128,
                },

                "latency_seconds": latency_seconds,
            }

            out_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

            out_file.flush()

            print(
                f"[{i}/50] {record['doc_id']} "
                f"({latency_seconds:.2f}s)"
            )

    print()
    print("Finished.")
    print("Saved responses to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()