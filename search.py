# Script for performing semantic search over the bBSARD corpus.
# bi-encoder retrieval
# Loads a fine-tuned SentenceTransformer model, a FAISS index,
# and corresponding metadata, then retrieves the most relevant
# legal documents for a given query.

import json
import faiss
from sentence_transformers import SentenceTransformer


def load_resources(model_name, index_path, metadata_path):
    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = data["documents"]

    return model, index, documents


def search(query, model, index, documents, top_k=5):    # change top_k for more retrievals
    # encode query
    emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(emb)

    # retriever
    scores, indices = index.search(emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append((documents[int(idx)], float(score)))

    return results


def main():
    # config
    model_name = "Arash-Alborz/minilm-dutch-legal-retrieval"
    index_path = "faiss_index.bin"
    metadata_path = "faiss_metadata.json"

    model, index, documents = load_resources(
        model_name, index_path, metadata_path
    )

    # example query
    query = "Wat gebeurt er met een huis na een scheiding?"    # en: "What happens to a house after a divorce?"

    print(f"\nQuery: {query}\n")

    results = search(query, model, index, documents)

    for i, (text, score) in enumerate(results, 1):
        print(f"Result {i} (score={score:.4f})\n{text}\n")


if __name__ == "__main__":
    main()
