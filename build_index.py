# Script to build a FAISS index for semantic retrieval over the bBSARD dataset.

# This script loads a fine-tuned SentenceTransformer model, encodes all legal
# articles from the bBSARD corpus, and builds a FAISS index for efficient
# similarity search. The index and corresponding metadata are saved locally.

# Note: Encoding the full corpus may require significant memory and is faster on GPU.

import json
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def main():
    # config
    model_path = "Arash-Alborz/minilm-dutch-legal-retrieval" 
    index_path = "faiss_index.bin"
    metadata_path = "faiss_metadata.json"
    batch_size = 32

    model = SentenceTransformer(model_path)

    corpus = load_dataset("clips/bBSARD", "corpus")["nl"]
    corpus_ids = corpus["id"]
    documents = corpus["article"]

    # corpus encoding
    doc_embeddings = model.encode(
        documents,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # normalizing embeddings (for cosine similarity)
    faiss.normalize_L2(doc_embeddings)

    # building index
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)

    faiss.write_index(index, index_path)

    # metadata
    metadata = {
        "corpus_ids": [str(i) for i in corpus_ids],
        "documents": documents
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
