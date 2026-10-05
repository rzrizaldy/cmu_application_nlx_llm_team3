import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


development_file = "data/pittsburgh311/dataset_splits/development_dataset.jsonl"


development_records = []

with open(development_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip() != "":
            record = json.loads(line)

            # Only use text records
            if record.get("raw_text"):
                development_records.append(record)


development_texts = []

for record in development_records:
    development_texts.append(record["raw_text"])


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

development_vectors = vectorizer.fit_transform(development_texts)


def retrieve_similar_requests(query, top_k=3):
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        development_vectors
    )[0]

    ranked_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        record = development_records[index]
        metadata = record["metadata"]

        result = {
            "similarity": float(similarities[index]),
            "example_request": record["raw_text"],
            "request_type_id": metadata.get("request_type_id"),
            "issue": metadata.get("issue"),
            "category": metadata.get("category"),
            "department": metadata.get("department")
        }

        results.append(result)

    return results


if __name__ == "__main__":
    test_query = "A new city tree should be planted near this street."

    results = retrieve_similar_requests(test_query)

    print("Query:")
    print(test_query)
    print()

    for i, result in enumerate(results, start=1):
        print(f"Result {i}")
        print(result)
        print()