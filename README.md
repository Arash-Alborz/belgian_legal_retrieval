# Multilingual Semantic Search over Belgian Legal Texts

This project implements a semantic search system over Belgian statutory law articles (bBSARD dataset) using dense retrieval, FAISS indexing, and cross-encoder reranking. The system supports multilingual queries (English, German, Dutch) through automatic translation.
The bBSARD dataset includes both Dutch and French, however for this project only the Dutch subset was used.

---

## Demo

A live demo of the system is available on Hugging Face Spaces:

https://huggingface.co/spaces/Arash-Alborz/dutch-legal-search

---

## Overview

The pipeline consists of three main components:

1. **Dense Retrieval (Bi-Encoder)**  
   A fine-tuned SentenceTransformer model maps queries and documents into a shared embedding space.
   Base model [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) shows weak retrieval, however fine-tuning improved the retrieval.
   Although the model is very small, it shows promissing retrieval abilities (see evaluation).

3. **FAISS Indexing**  
   Efficient similarity search over document embeddings using cosine similarity.

4. **Reranking (Cross-Encoder)**  
A cross-encoder model, [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2), re-ranks retrieved candidates for improved ranking quality.

5. **Multilingual Support**  
   Queries in English and German are translated into Dutch before retrieval, and results are translated back into the original language.

---

## Dataset

- **bBSARD** (Belgian Statutory Article Retrieval Dataset)  
- Source: https://huggingface.co/datasets/clips/bBSARD  
- Language: Dutch 
- Size: ~22,417 legal articles
- The data consists of a corpus (with the body of law articles) and train/test sets with legal questions (queries) and their relevant articles.
- For the fine-tuning, the train set of 886 legal questions were split into train and validation. The test set of 222 legal queries were only used for evaluation.

---
- **Fine-tuning was done by contrastive loss for the training and cosine similarity for validation set.**

- Evaluated with InformationRetrievalEvaluator

| Metric                | Value  |
|----------------------|--------|
| cosine_accuracy@1    | 0.2416 |
| cosine_accuracy@3    | 0.3652 |
| cosine_accuracy@5    | 0.4270 |
| cosine_accuracy@10   | 0.5449 |
| cosine_precision@1   | 0.2416 |
| cosine_precision@3   | 0.1910 |
| cosine_precision@5   | 0.1697 |
| cosine_precision@10  | 0.1489 |
| cosine_recall@1      | 0.1034 |
| cosine_recall@3      | 0.1819 |
| cosine_recall@5      | 0.2372 |
| cosine_recall@10     | 0.3664 |
| cosine_ndcg@10       | 0.3122 |
| cosine_mrr@10        | 0.3263 |
| cosine_map@100       | 0.2787 |

---

- **Retrieval evaluation**
- Followingis the evaluation of the retrieval with the final fine-tuned model.

| Metric    | Value  |
|-----------|--------|
| R@100     | 0.5859 |
| R@200     | 0.6425 |
| MRR@100   | 0.3001 |
| MAP@100   | 0.2027 |
| nDCG@10   | 0.2511 |
| nDCG@100  | 0.3203 |


## Model

- Fine-tuned model:  
  [Arash-Alborz/minilm-dutch-legal-retrieval](https://huggingface.co/Arash-Alborz/minilm-dutch-legal-retrieval)

- Base model:  
  [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

- Reranker:  
  [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)

- Translation models (Hugging Face):
  - [Helsinki-NLP/opus-mt-en-nl](https://huggingface.co/Helsinki-NLP/opus-mt-en-nl)  
  - [Helsinki-NLP/opus-mt-de-nl](https://huggingface.co/Helsinki-NLP/opus-mt-de-nl)  
  - [Helsinki-NLP/opus-mt-nl-en](https://huggingface.co/Helsinki-NLP/opus-mt-nl-en)  
  - [Helsinki-NLP/opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de)

---

## Repository Structure

fine_tune.py              # Fine-tuning the retriever model
build_index.py            # Build FAISS index from corpus
search.py                 # Retrieval only (FAISS + embeddings)
rerank_search.py          # Retrieval + reranking
multilingual_search.py    # Full pipeline (translation + reranking)
requirements.txt          # Dependencies

---

## Installation

Clone the repository and install dependencies:

pip install -r requirements.txt

---

## Usage

### 1. Build FAISS index

python build_index.py

This will create:

- faiss_index.bin
- faiss_metadata.json

---

### 2. Run retrieval

#### Basic retrieval:

python search.py

#### Retrieval + reranking:

python rerank_search.py

#### Multilingual search:

python multilingual_search.py

---

## Example Query
Since the legal questions in bBSARD are asked by laypeople, the queries can also be in an ordinary manner and not with strict legal terminology, such as:

NL: Wat gebeurt er met een huis na een scheiding?

or 

EN: What happens to a house after a divorce?

or

DE: Was passiert mit einem Haus nach einer Scheidung?

---

## Notes

- Reranking improves accuracy but increases latency.
- Translation enables multilingual access but adds additional computation time.
- FAISS retrieval is fast and scales well to large datasets.

---

## Limitations

- Translation quality may affect retrieval accuracy.
- The system operates on full articles (no chunking).
- Performance depends on CPU/GPU availability.

---

## Future Work

- Add document chunking for finer-grained retrieval (250-tokens windows with 50-tokens overlap)
- Improve translation efficiency
- Deploy as an API service in a full RAG pipeline for legal QA

---
## Dependencies and Licenses

This project uses the following external models and datasets:

- sentence-transformers/all-MiniLM-L6-v2  
- cross-encoder/ms-marco-MiniLM-L-6-v2  
- Helsinki-NLP translation models  
- bBSARD dataset  

These components are subject to their respective licenses on Hugging Face.

## Author

**Arash Alborz**  
- GitHub: https://github.com/Arash-Alborz  
- Hugging Face: https://huggingface.co/Arash-Alborz  
- Project: Multilingual Legal Semantic Search
