# Multilingual semantic search with reranking over the bBSARD dataset.

# This script extends the retrieval pipeline with multilingual support:
# 1. Detects query language (EN, DE, NL)
# 2. Translates non-Dutch queries into Dutch for retrieval
# 3. Performs dense retrieval using SentenceTransformer + FAISS
# 4. Reranks results using a CrossEncoder
# 5. Translates results back to the original query language

# This allows users to query Dutch legal documents also in English or German.

# Note: Dutch queries do not undergo any translation.

import json
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect


# en → nl
tok_en_nl = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl")
mod_en_nl = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-nl")

# de → nl
tok_de_nl = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-nl")
mod_de_nl = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-nl")

# nl → en
tok_nl_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-nl-en")
mod_nl_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-nl-en")

# en → de
tok_en_de = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
mod_en_de = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")


# translation helpers
def translate(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


def detect_language(query):
    try:
        return detect(query)
    except:
        return "nl"


def translate_query(query, lang):
    if lang == "en":
        return translate([query], tok_en_nl, mod_en_nl)[0]
    elif lang == "de":
        return translate([query], tok_de_nl, mod_de_nl)[0]
    return query


def translate_back(texts, lang):
    if lang == "en":
        return translate(texts, tok_nl_en, mod_nl_en)

    elif lang == "de":
        # nl → en
        en_texts = translate(texts, tok_nl_en, mod_nl_en)
        # en → de
        return translate(en_texts, tok_en_de, mod_en_de)

    return texts


# resource
def load_resources():
    model = SentenceTransformer("Arash-Alborz/minilm-dutch-legal-retrieval")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    index = faiss.read_index("faiss_index.bin")

    with open("faiss_metadata.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus_ids = data["corpus_ids"]
    documents = data["documents"]

    return model, reranker, index, corpus_ids, documents


# search (bi-encoder + cross-encoder)

def search(query, model, reranker, index, corpus_ids, documents,
           top_k=5, retrieval_k=100):

    # detect + translate query
    lang = detect_language(query)
    translated_query = translate_query(query, lang)

    # encode query
    emb = model.encode([translated_query], convert_to_numpy=True)
    faiss.normalize_L2(emb)

    # retrieval
    scores, indices = index.search(emb, retrieval_k)

    candidates = []
    for idx in indices[0]:
        candidates.append({
            "id": corpus_ids[int(idx)],
            "text": documents[int(idx)]
        })

    # rerank
    pairs = [(translated_query, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)

    for i, c in enumerate(candidates):
        c["score"] = float(rerank_scores[i])

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    results = candidates[:top_k]

    # batch back-translation (--> batch shows better and faster results))
    texts = [r["text"] for r in results]
    translated_texts = translate_back(texts, lang)

    final_results = []
    for r, t in zip(results, translated_texts):
        final_results.append({
            "id": r["id"],
            "text": t,
            "score": r["score"]
        })

    return translated_query, final_results

def main():
    model, reranker, index, corpus_ids, documents = load_resources()

    query = "Was passiert mit einem Haus nach einer Scheidung?"    # or en: "What happens to a house after divorce?"

    print(f"\nOriginal query: {query}")

    translated_query, results = search(
        query, model, reranker, index, corpus_ids, documents
    )

    print(f"Translated query: {translated_query}\n")
    print("RESULTS:\n")

    for i, r in enumerate(results, 1):
        print(f"Result {i} (score={r['score']:.4f})")
        print(r["id"])
        print(r["text"][:200])
        print("-" * 50)


if __name__ == "__main__":
    main()
