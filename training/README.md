# Difficulty classifier training

Everything in this folder is written to run in **Google Colab**, not on the
machine running the API server. Fine-tuning a transformer, even a small
one, is exactly the heavy local ML job this project avoids -- `app/`
never requires torch/transformers to be installed; it works in pure
heuristic mode (`app/router/heuristic.py`) until a checkpoint shows up.

## Files

- `prepare_dataset.py` -- builds the labeled dataset (MMLU + GSM8K +
  hand-written easy examples) as JSONL. See its docstring for the labeling
  strategy and its limitations.
- `train_classifier.py` -- fine-tunes a small sequence-classification model
  (MiniLM by default) on that dataset using `transformers.Trainer`.
- `colab_finetune_difficulty_classifier.ipynb` -- runs both scripts
  end-to-end in Colab: install deps, upload the two scripts, build the
  dataset, train, sanity-check, zip and download the checkpoint.

## Workflow

1. Open `colab_finetune_difficulty_classifier.ipynb` in Google Colab
   (Runtime -> Change runtime type -> GPU).
2. Run the cells top to bottom. You'll be prompted to upload
   `prepare_dataset.py` and `train_classifier.py` partway through.
3. The last cell downloads `difficulty-classifier.zip`.
4. Unzip it locally into `training/checkpoints/difficulty-classifier/`
   (or wherever `classifier_checkpoint_dir` in `app/config.py` points).
5. Restart the API (or just let it be -- `DifficultyClassifier` checks for
   the checkpoint directory lazily on first request, no code change
   needed). It'll log that it loaded the classifier instead of falling
   back to the heuristic.

## Expected checkpoint layout

```
training/checkpoints/difficulty-classifier/
    config.json
    model.safetensors
    tokenizer_config.json
    vocab.txt (or equivalent, depending on base model)
    label_map.json          <- {"0": "easy", "1": "medium", "2": "hard"}
```

`label_map.json` is written by `train_classifier.py` and read (never
assumed) by `app/router/classifier.py` -- the label-to-index order is
whatever `sorted(set(labels))` produced during training, not hardcoded.

## If you skip this entirely

The API works without ever doing this -- `app/router/heuristic.py`'s
regex/word-count scorer serves every request instead. The benchmark
(`benchmark/`) will tell you whether training a real classifier is worth
the trouble for your traffic, by comparing router-with-heuristic against
router-with-classifier once you have one.
