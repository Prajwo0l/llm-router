"""
Fine-tunes a small sequence-classification model (default: MiniLM, a
lighter alternative to DistilBERT -- either works, MiniLM is faster to
train on a free Colab GPU) on the difficulty dataset from
prepare_dataset.py.

Meant to run in Colab (see training/README.md and the accompanying
notebook), NOT on this machine -- torch/transformers are exactly the
"heavy ML job" this project's local codebase avoids. Written and reviewed
here; run there.

Output layout (matches what app/router/classifier.py expects):
    <output_dir>/
        config.json, model.safetensors, tokenizer files...  (from Trainer.save_model)
        label_map.json   {"0": "easy", "1": "medium", "2": "hard"}

Copy that whole output_dir back into this repo at the path
classifier_checkpoint_dir points to (default:
training/checkpoints/difficulty-classifier) and the running API will pick
it up automatically -- DifficultyClassifier._load_model() checks for that
directory on first request.
"""
from __future__ import annotations

import argparse
import json
import os


def train(
    dataset_path: str,
    output_dir: str,
    base_model: str = "microsoft/MiniLM-L12-H384-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> None:
    import numpy as np
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    examples = []
    with open(dataset_path) as f:
        for line in f:
            examples.append(json.loads(line))

    labels = sorted({ex["label"] for ex in examples})  # deterministic order
    label_to_id = {label: i for i, label in enumerate(labels)}

    dataset = Dataset.from_list(
        [{"text": ex["text"], "label": label_to_id[ex["label"]]} for ex in examples]
    ).train_test_split(test_size=val_fraction, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    tokenized = dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=len(labels))

    def compute_metrics(eval_pred):
        logits, refs = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(refs, preds),
            "f1_macro": f1_score(refs, preds, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "_trainer_run"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=25,
        report_to=[],  # no wandb/etc by default -- keep the Colab run self-contained
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print(f"Final eval metrics: {eval_metrics}")

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    id_to_label = {str(i): label for label, i in label_to_id.items()}
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(id_to_label, f, indent=2)

    print(f"Saved classifier + label_map.json to {output_dir}")
    print("Copy this directory back into the repo at the path classifier_checkpoint_dir points to.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="training/data/difficulty_dataset.jsonl")
    parser.add_argument("--output-dir", type=str, default="training/checkpoints/difficulty-classifier")
    parser.add_argument("--base-model", type=str, default="microsoft/MiniLM-L12-H384-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
