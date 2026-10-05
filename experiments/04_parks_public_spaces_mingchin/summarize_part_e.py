results = [
    {
        "evaluation": "Baseline Eval 1",
        "full_accuracy": 0.00,
        "latency": 2.57
    },
    {
        "evaluation": "Baseline Eval 3",
        "full_accuracy": 0.00,
        "latency": 2.12
    },
    {
        "evaluation": "Structured Output",
        "full_accuracy": 0.00,
        "latency": 5.69
    },
    {
        "evaluation": "Tool Calling",
        "full_accuracy": 0.50,
        "latency": 7.61
    },
    {
        "evaluation": "TF-IDF Top 1",
        "full_accuracy": 0.54,
        "latency": 0.0004
    },
    {
        "evaluation": "TF-IDF Majority Vote",
        "full_accuracy": 0.68,
        "latency": 0.0004
    }
]


print("Part E Performance Summary")
print("--------------------------")

for result in results:
    print(
        f"{result['evaluation']}: "
        f"accuracy={result['full_accuracy'] * 100:.1f}%, "
        f"latency={result['latency']:.4f}s"
    )


print()
print("Guardrail")
print("---------")
print("Catch rate: 91.7%")
print("Over-refusal rate: 0.0%")
print("Human agreement: 91.7%")
print("Weakest category: Out of scope (66.7%)")
print("Added tokens per request: 0")
print("Added latency: no measurable overhead")