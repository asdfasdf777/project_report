import ast
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from config import (
    SPLITS_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    BERT_MODEL_NAME,
    MAX_TEXT_LENGTH,
    NUM_LABELS,
    GNN_HIDDEN_DIM,
    LEARNING_RATE,
    EPOCHS,
    RANDOM_SEED
)

from fusion_dataset import FusionDataset
from cross_attention_model import (
    CrossAttentionFusionModel
)

from baseline_utils import calculate_metrics
from utils import set_seed


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def single_item_collate(batch):
    return batch[0]


def evaluate(
    model,
    loader
):

    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for batch in loader:

            input_ids = (
                batch["input_ids"]
                .unsqueeze(0)
                .to(DEVICE)
            )

            attention_mask = (
                batch["attention_mask"]
                .unsqueeze(0)
                .to(DEVICE)
            )

            graph_x = (
                batch["graph_x"]
                .to(DEVICE)
            )

            edge_index = (
                batch["edge_index"]
                .to(DEVICE)
            )

            target = (
                batch["labels"]
                .unsqueeze(0)
                .to(DEVICE)
            )

            logits = model(
                input_ids,
                attention_mask,
                graph_x,
                edge_index
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.append(
                probs.cpu().numpy()[0]
            )

            labels.append(
                target.cpu().numpy()[0]
            )

    probabilities = np.asarray(
        probabilities
    )

    labels = np.asarray(
        labels
    )

    return calculate_metrics(
        probabilities,
        labels
    )


def main():

    set_seed(RANDOM_SEED)

    print(
        "Device:",
        DEVICE
    )

    dataframe = pd.read_csv(
        SPLITS_DIR
        / "context_dataset.csv"
    )

    train_df = dataframe[
        dataframe["split"] == "training"
    ].reset_index(drop=True)

    val_df = dataframe[
        dataframe["split"] == "validation"
    ].reset_index(drop=True)

    test_df = dataframe[
        dataframe["split"] == "test"
    ].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_NAME
    )

    graph_directory = (
        PROCESSED_DIR / "graphs"
    )

    train_dataset = FusionDataset(
        train_df,
        tokenizer,
        graph_directory,
        MAX_TEXT_LENGTH
    )

    val_dataset = FusionDataset(
        val_df,
        tokenizer,
        graph_directory,
        MAX_TEXT_LENGTH
    )

    test_dataset = FusionDataset(
        test_df,
        tokenizer,
        graph_directory,
        MAX_TEXT_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=single_item_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=single_item_collate
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=single_item_collate
    )

    model = CrossAttentionFusionModel(
        BERT_MODEL_NAME,
        NUM_LABELS,
        graph_input_dim=25,
        graph_hidden_dim=GNN_HIDDEN_DIM
    )

    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_f1 = -1.0

    history = []

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            input_ids = (
                batch["input_ids"]
                .unsqueeze(0)
                .to(DEVICE)
            )

            attention_mask = (
                batch["attention_mask"]
                .unsqueeze(0)
                .to(DEVICE)
            )

            graph_x = (
                batch["graph_x"]
                .to(DEVICE)
            )

            edge_index = (
                batch["edge_index"]
                .to(DEVICE)
            )

            labels = (
                batch["labels"]
                .unsqueeze(0)
                .to(DEVICE)
            )

            optimizer.zero_grad()

            logits = model(
                input_ids,
                attention_mask,
                graph_x,
                edge_index
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(train_loader)
        )

        val_metrics = evaluate(
            model,
            val_loader
        )

        history.append({
            "epoch": epoch + 1,
            "loss": average_loss,
            **val_metrics
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"Loss={average_loss:.4f} "
            f"Val Macro-F1="
            f"{val_metrics['macro_f1']:.4f} "
            f"Val Micro-F1="
            f"{val_metrics['micro_f1']:.4f}"
        )

        if (
            val_metrics["macro_f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                val_metrics["macro_f1"]
            )

            MODELS_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                MODELS_DIR
                / "cross_attention_model.pt"
            )

    print(
        "\nLoading best cross-attention model..."
    )

    model.load_state_dict(
        torch.load(
            MODELS_DIR
            / "cross_attention_model.pt",
            map_location=DEVICE,
            weights_only=True
        )
    )

    test_metrics = evaluate(
        model,
        test_loader
    )

    print(
        "\nCROSS-ATTENTION TEST RESULTS"
    )

    print(
        f"Macro-F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Micro-F1: "
        f"{test_metrics['micro_f1']:.4f}"
    )

    print(
        f"Macro AUC-PR: "
        f"{test_metrics['macro_auc_pr']:.4f}"
    )

    print(
        f"Micro AUC-PR: "
        f"{test_metrics['micro_auc_pr']:.4f}"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_dir = (
        RESULTS_DIR / "predictions"
    )

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Generate final predictions.
    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for batch in test_loader:

            input_ids = (
                batch["input_ids"]
                .unsqueeze(0)
                .to(DEVICE)
            )

            attention_mask = (
                batch["attention_mask"]
                .unsqueeze(0)
                .to(DEVICE)
            )

            graph_x = (
                batch["graph_x"]
                .to(DEVICE)
            )

            edge_index = (
                batch["edge_index"]
                .to(DEVICE)
            )

            logits = model(
                input_ids,
                attention_mask,
                graph_x,
                edge_index
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.append(
                probs.cpu().numpy()[0]
            )

            labels.append(
                batch["labels"].numpy()
            )

    probabilities = np.asarray(
        probabilities
    )

    labels = np.asarray(
        labels
    )

    prediction_path = (
        predictions_dir
        / "cross_attention_predictions.npz"
    )

    np.savez(
        prediction_path,
        probabilities=probabilities,
        labels=labels,
        track_ids=test_df[
            "track_id"
        ].to_numpy()
    )

    result = {
        "model": "cross_attention",
        "num_labels": NUM_LABELS,
        "metrics": test_metrics,
        "history": history
    }

    result_path = (
        RESULTS_DIR
        / "cross_attention_results.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            indent=4
        )

    print(
        "\nSaved:",
        result_path
    )

    print(
        "Saved:",
        prediction_path
    )


if __name__ == "__main__":
    main()