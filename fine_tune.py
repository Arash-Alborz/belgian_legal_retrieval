import random
from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator


def load_data():
    corpus = load_dataset("clips/bBSARD", "corpus")["nl"]
    train_full = load_dataset("clips/bBSARD", "train")["nl"]

    corpus_ids = corpus["id"]
    corpus_texts = corpus["article"]

    id_to_text = {
        str(corpus_ids[i]): corpus_texts[i]
        for i in range(len(corpus_ids))
    }

    return train_full, corpus_ids, corpus_texts, id_to_text


def split_data(train_full, split_ratio=0.8):
    indices = list(range(len(train_full["id"])))
    random.shuffle(indices)

    split = int(split_ratio * len(indices))
    return indices[:split], indices[split:]


def build_train_examples(train_full, train_idx, id_to_text):
    train_examples = []

    for i in train_idx:
        query = train_full["question"][i]
        positives = [p.strip() for p in train_full["article_ids"][i].split(",")]

        for pos_id in positives:
            if pos_id in id_to_text:
                train_examples.append(
                    InputExample(texts=[query, id_to_text[pos_id]])
                )

    return train_examples


def build_evaluator(train_full, dev_idx, corpus_ids, corpus_texts):
    queries = {}
    relevant_docs = {}

    for i in dev_idx:
        qid = str(train_full["id"][i])
        queries[qid] = train_full["question"][i]

        relevant_docs[qid] = set(
            [p.strip() for p in train_full["article_ids"][i].split(",")]
        )

    corpus_dict = {
        str(corpus_ids[i]): corpus_texts[i]
        for i in range(len(corpus_ids))
    }

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus_dict,
        relevant_docs=relevant_docs,
        show_progress_bar=True
    )

    return evaluator


def main():
    # config
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    output_path = "models/minilm_finetuned"
    batch_size = 22
    epochs = 10

    # load model
    model = SentenceTransformer(model_name)

    # load data
    train_full, corpus_ids, corpus_texts, id_to_text = load_data()

    # split
    train_idx, dev_idx = split_data(train_full)

    # build training data
    train_examples = build_train_examples(train_full, train_idx, id_to_text)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # evaluator
    evaluator = build_evaluator(train_full, dev_idx, corpus_ids, corpus_texts)

    # loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # training
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=len(train_dataloader),
        warmup_steps=int(len(train_dataloader) * 0.1),
        show_progress_bar=True,
        output_path=output_path,
        save_best_model=True
    )


if __name__ == "__main__":
    main()
