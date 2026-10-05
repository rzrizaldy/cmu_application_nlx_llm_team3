import json

from omegaconf import OmegaConf

from src.generation import GenerationManager


files = {
    "Eval 3 TF-IDF + LLM":
        "outputs/tfidf_llm_eval3_responses.jsonl",

    "Eval 4 TF-IDF Majority Vote + LLM":
        "outputs/tfidf_vote_llm_eval4_responses.jsonl"
}


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
    }
})


generator = GenerationManager()

print("Loading tokenizer...")

model, tokenizer, device = generator._load_model_and_tokenizer(cfg)

print("Tokenizer loaded.")
print()


def count_tokens(text):
    return len(
        tokenizer(
            text,
            add_special_tokens=False
        )["input_ids"]
    )


for name, path in files.items():

    input_tokens = []
    output_tokens = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            # Count the resident request as input
            # plus the retrieved context used for that evaluation.

            if "retrieved_example" in record:

                retrieved = record["retrieved_example"]

                prompt_text = f"""
Request:
{record["input"]}

Retrieved example:
{retrieved}
"""

            elif "top_3_results" in record:

                prompt_text = f"""
Request:
{record["input"]}

Top 3 retrieved examples:
{record["top_3_results"]}

Majority result:
{record["majority_result"]}
"""

            else:
                prompt_text = record["input"]

            response_text = record["model_response"]

            input_tokens.append(
                count_tokens(prompt_text)
            )

            output_tokens.append(
                count_tokens(response_text)
            )


    avg_input = sum(input_tokens) / len(input_tokens)
    avg_output = sum(output_tokens) / len(output_tokens)
    avg_total = avg_input + avg_output


    print(name)
    print("-" * len(name))

    print(
        f"Average Input Tokens: "
        f"{avg_input:.1f}"
    )

    print(
        f"Average Output Tokens: "
        f"{avg_output:.1f}"
    )

    print(
        f"Average Total Tokens: "
        f"{avg_total:.1f}"
    )

    print()