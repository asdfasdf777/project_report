import ast
import json

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config import (
    BERT_MODEL_NAME,
    MAX_TEXT_LENGTH,
    NUM_LABELS,
    SPLITS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    LEARNING_RATE,
    EPOCHS
)

from bert_model import BERTTagClassifier
from utils import get_device, set_seed


class ContextTextDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=128
    ):

        self.dataframe = (
            dataframe.reset_index(drop=True)
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        text = str(row["text"])

        labels = torch.tensor(
            ast.literal_eval(
                row["labels"]
            ),
            dtype=torch.float32
        )

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "labels":
                labels
        }


def calculate_f1(
    predictions,
    labels
):

    probabilities = torch.sigmoid(
        predictions
    )

    predicted = (
        probabilities >= 0.5
    ).float()

    true_positive = (
        predicted * labels
    ).sum(dim=0)

    false_positive = (
        predicted * (1 - labels)
    ).sum(dim=0)

    false_negative = (
        (1 - predicted) * labels
    ).sum(dim=0)

    precision = (
        true_positive
        /
        (
            true_positive
            + false_positive
            + 1e-8
        )
    )

    recall = (
        true_positive
        /
        (
            true_positive
            + false_negative
            + 1e-8
        )
    )

    f1 = (
        2 * precision * recall
        /
        (
            precision
            + recall
            + 1e-8
        )
    )

    macro_f1 = (
        f1.mean().item()
    )

    total_tp = true_positive.sum()
    total_fp = false_positive.sum()
    total_fn = false_negative.sum()

    micro_precision = (
        total_tp
        /
        (
            total_tp
            + total_fp
            + 1e-8
        )
    )

    micro_recall = (
        total_tp
        /
        (
            total_tp
            + total_fn
            + 1e-8
        )
    )

    micro_f1 = (
        2
        * micro_precision
        * micro_recall
        /
        (
            micro_precision
            + micro_recall
            + 1e-8
        )
    )

    return (
        macro_f1,
        micro_f1.item()
    )


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

            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )

            logits = model(
                input_ids,
                attention_mask
            )

            all_predictions.append(
                logits.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    predictions = torch.cat(
        all_predictions,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    return calculate_f1(
        predictions,
        labels
    )


def main():

    print(
        "Starting context BERT training..."
    )

    set_seed(42)

    device = get_device()

    print(
        "Device:",
        device
    )

    dataset_path = (
        SPLITS_DIR
        / "context_dataset.csv"
    )

    dataframe = pd.read_csv(
        dataset_path
    )

    train_df = dataframe[
        dataframe["split"] == "training"
    ].copy()

    val_df = dataframe[
        dataframe["split"] == "validation"
    ].copy()

    test_df = dataframe[
        dataframe["split"] == "test"
    ].copy()

    print(
        "Dataset samples:",
        len(dataframe)
    )

    print(
        "Train:",
        len(train_df)
    )

    print(
        "Validation:",
        len(val_df)
    )

    print(
        "Test:",
        len(test_df)
    )

    print(
        "\nLoading BERT tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_NAME
    )

    train_dataset = ContextTextDataset(
        train_df,
        tokenizer,
        MAX_TEXT_LENGTH
    )

    val_dataset = ContextTextDataset(
        val_df,
        tokenizer,
        MAX_TEXT_LENGTH
    )

    test_dataset = ContextTextDataset(
        test_df,
        tokenizer,
        MAX_TEXT_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )

    print(
        "Creating BERT model..."
    )

    model = BERTTagClassifier(
        BERT_MODEL_NAME,
        NUM_LABELS
    )

    model = model.to(device)

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = {
        "train_loss": [],
        "val_macro_f1": [],
        "val_micro_f1": []
    }

    print(
        "\nBeginning training..."
    )

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )

            optimizer.zero_grad()

            logits = model(
                input_ids,
                attention_mask
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

        average_loss = (
            total_loss
            /
            len(train_loader)
        )

        val_macro, val_micro = evaluate(
            model,
            val_loader,
            device
        )

        history["train_loss"].append(
            average_loss
        )

        history["val_macro_f1"].append(
            val_macro
        )

        history["val_micro_f1"].append(
            val_micro
        )

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Average Loss={average_loss:.4f}"
        )

        print(
            f"Validation Macro-F1={val_macro:.4f}"
        )

        print(
            f"Validation Micro-F1={val_micro:.4f}"
        )

    print(
        "\nEvaluating on test set..."
    )

    test_macro, test_micro = evaluate(
        model,
        test_loader,
        device
    )

    print(
        "\nFINAL CONTEXT BERT TEST RESULTS"
    )

    print(
        f"Macro-F1: {test_macro}"
    )

    print(
        f"Micro-F1: {test_micro}"
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        MODELS_DIR
        / "context_bert_model.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    results = {
        "test_macro_f1": test_macro,
        "test_micro_f1": test_micro,
        "history": history
    }

    results_path = (
        RESULTS_DIR
        / "context_bert_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        "\nContext BERT model saved to:",
        model_path
    )

    print(
        "Context BERT results saved to:",
        results_path
    )


if __name__ == "__main__":
    main()