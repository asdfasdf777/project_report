import json

import pandas as pd
import torch

from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from config import (
    SPLITS_DIR,
    BERT_MODEL_NAME,
    MAX_TEXT_LENGTH,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    MODELS_DIR,
    RESULTS_DIR
)

from bert_dataset import MusicTextDataset
from bert_model import BERTTagClassifier
from utils import set_seed, get_device


def calculate_f1(predictions, labels):

    predictions = predictions.int()
    labels = labels.int()

    true_positive = (
        predictions * labels
    ).sum(dim=0).float()

    false_positive = (
        predictions * (1 - labels)
    ).sum(dim=0).float()

    false_negative = (
        (1 - predictions) * labels
    ).sum(dim=0).float()

    precision = (
        true_positive /
        (true_positive + false_positive + 1e-8)
    )

    recall = (
        true_positive /
        (true_positive + false_negative + 1e-8)
    )

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-8)
    )

    macro_f1 = f1.mean().item()

    total_tp = true_positive.sum()
    total_fp = false_positive.sum()
    total_fn = false_negative.sum()

    micro_precision = (
        total_tp /
        (total_tp + total_fp + 1e-8)
    )

    micro_recall = (
        total_tp /
        (total_tp + total_fn + 1e-8)
    )

    micro_f1 = (
        2 * micro_precision * micro_recall /
        (micro_precision + micro_recall + 1e-8)
    )

    return macro_f1, micro_f1.item()


def evaluate(
    model,
    loader,
    device
):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids,
                attention_mask
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            )

            all_predictions.append(
                predictions.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    predictions = torch.cat(
        all_predictions
    )

    labels = torch.cat(
        all_labels
    )

    return calculate_f1(
        predictions,
        labels
    )


def main():

    set_seed(42)

    print("Starting BERT training script...", flush=True)

    device = get_device()

    print("Device:", device, flush=True)

    print("Loading dataset...", flush=True)

    dataframe = pd.read_csv(
        SPLITS_DIR / "dataset.csv"
    )

    print(
        "Dataset loaded:",
        len(dataframe),
        "rows",
        flush=True
    )

    train_df = dataframe[
        dataframe["split"] == "training"
    ]

    val_df = dataframe[
        dataframe["split"] == "validation"
    ]

    test_df = dataframe[
        dataframe["split"] == "test"
    ]

    print(
        "Train:",
        len(train_df),
        flush=True
    )

    print(
        "Validation:",
        len(val_df),
        flush=True
    )

    print(
        "Test:",
        len(test_df),
        flush=True
    )

    print("Loading labels...", flush=True)

    with open(
        SPLITS_DIR / "labels.json",
        "r",
        encoding="utf-8"
    ) as f:

        labels = json.load(f)

    print(
        "Number of labels:",
        len(labels),
        flush=True
    )

    print("Loading BERT tokenizer...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_NAME
    )

    print("Tokenizer loaded.", flush=True)

    print("Creating datasets...", flush=True)

    train_dataset = MusicTextDataset(
        train_df,
        tokenizer,
        MAX_TEXT_LENGTH
    )

    val_dataset = MusicTextDataset(
        val_df,
        tokenizer,
        MAX_TEXT_LENGTH
    )

    test_dataset = MusicTextDataset(
        test_df,
        tokenizer,
        MAX_TEXT_LENGTH
    )

    print("Datasets created.", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )

    print("Loading BERT model...", flush=True)

    model = BERTTagClassifier(
        BERT_MODEL_NAME,
        len(labels)
    )

    print("BERT model loaded.", flush=True)

    model.to(device)

    print(
        "Model moved to:",
        device,
        flush=True
    )

    loss_function = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("Beginning training...", flush=True)

    history = []

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels_batch = batch[
                "labels"
            ].to(device)

            logits = model(
                input_ids,
                attention_mask
            )

            loss = loss_function(
                logits,
                labels_batch
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss /
            len(train_loader)
        )

        val_macro_f1, val_micro_f1 = evaluate(
            model,
            val_loader,
            device
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Loss: {average_loss:.4f}"
        )

        print(
            f"Validation Macro-F1: {val_macro_f1:.4f}"
        )

        print(
            f"Validation Micro-F1: {val_micro_f1:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "loss": average_loss,
            "macro_f1": val_macro_f1,
            "micro_f1": val_micro_f1
        })

    test_macro_f1, test_micro_f1 = evaluate(
        model,
        test_loader,
        device
    )

    print("\nFINAL TEST RESULTS")

    print(
        "Macro-F1:",
        test_macro_f1
    )

    print(
        "Micro-F1:",
        test_micro_f1
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "bert_model.pt"

    torch.save(
        model.state_dict(),
        model_path
    )

    history_path = RESULTS_DIR / "bert_history.json"

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    print("\nModel saved to:", model_path)
    print("Training history saved to:", history_path)


if __name__ == "__main__":
    main()