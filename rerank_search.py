# Script for "semantic search + reranking" over the bBSARD dataset.

# This script performs a two-stage retrieval pipeline:
# 1. Dense retrieval using a fine-tuned SentenceTransformer model and FAISS.
# 2. Reranking of top retrieved documents using CrossEncoder.

# The retriever selects the top-k candidate documents based on vector
# similarity. These candidates are then reranked using a cross-encoder to produce more accurate final results.

import json
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


def load_resources(model_name, index_path, metadata_path, reranker_name):
    # retriever + FAISS index + metadata
    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus_ids = data["corpus_ids"]
    documents = data["documents"]

    # reranker
    reranker = CrossEncoder(reranker_name)

    return model, index, corpus_ids, documents, reranker


def search(query, model, index, corpus_ids, documents, reranker,
           top_k=5, retrieval_k=100):

    # query encoding
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # FAISS retrieval (top retrieval_k candidates --> change retrieval_k for more retrieved articles to feed in reranker)
    scores, indices = index.search(query_embedding, retrieval_k)

    candidates = []
    for idx in indices[0]:
        candidates.append({
            "id": corpus_ids[int(idx)],
            "text": documents[int(idx)]
        })

    # query-document pairs for reranker
    pairs = [(query, c["text"]) for c in candidates]

    # reranking
    rerank_scores = reranker.predict(pairs)

    # attach scores
    for i, c in enumerate(candidates):
        c["score"] = float(rerank_scores[i])

    # sorting by reranker result
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    # return top_k results (--> change top_k to change number of top ranks)
    return candidates[:top_k]


def main():
    # config
    model_name = "Arash-Alborz/minilm-dutch-legal-retrieval"
    reranker_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    index_path = "faiss_index.bin"
    metadata_path = "faiss_metadata.json"

    # load resources
    model, index, corpus_ids, documents, reranker = load_resources(
        model_name, index_path, metadata_path, reranker_name
    )

    # example query (Dutch)
    query = "Wat gebeurt er met een huis na een scheiding?"    # en: "What happens to a hous after divorce?"

    print(f"\nQuery: {query}\n")

    results = search(query, model, index, corpus_ids, documents, reranker)

    for i, r in enumerate(results, 1):
        print(f"Result {i} (score={r['score']:.4f})")
        print(r["id"])
        print(r["text"][:200])
        print("-" * 50)


if __name__ == "__main__":
    main()
