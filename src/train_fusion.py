import json

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import (
    SPLITS_DIR,
    PROCESSED_DIR,
    BERT_MODEL_NAME,
    MAX_TEXT_LENGTH,
    FUSION_HIDDEN_DIM,
    GNN_HIDDEN_DIM,
    NUM_LABELS,
    MODELS_DIR,
    RESULTS_DIR
)

from fusion_dataset import FusionDataset
from fusion_model import FusionModel


def calculate_f1(
    predictions,
    labels,
    threshold=0.5
):

    predictions = (
        torch.sigmoid(predictions)
        >= threshold
    ).float()

    labels = labels.float()

    true_positive = (
        predictions * labels
    ).sum(dim=0)

    false_positive = (
        predictions * (1 - labels)
    ).sum(dim=0)

    false_negative = (
        (1 - predictions) * labels
    ).sum(dim=0)

    precision = (
        true_positive /
        (
            true_positive +
            false_positive +
            1e-8
        )
    )

    recall = (
        true_positive /
        (
            true_positive +
            false_negative +
            1e-8
        )
    )

    f1 = (
        2 * precision * recall /
        (
            precision +
            recall +
            1e-8
        )
    )

    macro_f1 = f1.mean()

    total_tp = true_positive.sum()
    total_fp = false_positive.sum()
    total_fn = false_negative.sum()

    micro_precision = (
        total_tp /
        (
            total_tp +
            total_fp +
            1e-8
        )
    )

    micro_recall = (
        total_tp /
        (
            total_tp +
            total_fn +
            1e-8
        )
    )

    micro_f1 = (
        2 * micro_precision *
        micro_recall /
        (
            micro_precision +
            micro_recall +
            1e-8
        )
    )

    return (
        macro_f1.item(),
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
                .unsqueeze(0)
                .to(device)
            )

            attention_mask = (
                batch["attention_mask"]
                .unsqueeze(0)
                .to(device)
            )

            graph_x = (
                batch["graph_x"]
                .to(device)
            )

            edge_index = (
                batch["edge_index"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .unsqueeze(0)
                .to(device)
            )

            logits = model(
                input_ids,
                attention_mask,
                graph_x,
                edge_index
            )

            all_predictions.append(
                logits.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    if len(all_predictions) == 0:

        return 0.0, 0.0

    predictions = torch.cat(
        all_predictions,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    print(
        "Evaluation shapes:",
        "predictions =",
        tuple(predictions.shape),
        "labels =",
        tuple(labels.shape)
    )

    return calculate_f1(
        predictions,
        labels
    )


def main():

    print(
        "Starting GNN-BERT fusion training..."
    )

    # -------------------------
    # Device
    # -------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        "Device:",
        device
    )

    # -------------------------
    # Dataset
    # -------------------------

    dataset_path = (
        SPLITS_DIR /
        "context_dataset.csv"
    )

    dataframe = pd.read_csv(
        dataset_path
    )

    print(
        "Dataset samples:",
        len(dataframe)
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

    # -------------------------
    # Tokenizer
    # -------------------------

    print(
        "\nLoading BERT tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_NAME
    )

    print(
        "Tokenizer loaded."
    )

    # -------------------------
    # Dataset objects
    # -------------------------

    graph_directory = (
        PROCESSED_DIR /
        "graphs"
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

    # -------------------------
    # DataLoaders
    # -------------------------

    def single_item_collate(batch):
       return batch[0]

    

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

    print(
        "Fusion datasets created."
    )

    # -------------------------
    # Model
    # -------------------------

    print(
        "\nCreating fusion model..."
    )

    model = FusionModel(
        bert_name=BERT_MODEL_NAME,
        num_labels=NUM_LABELS,
        graph_input_dim=25,
        graph_hidden_dim=GNN_HIDDEN_DIM
    )

    model = model.to(device)

    print(
        "Fusion model created."
    )

    # -------------------------
    # Training
    # -------------------------

    loss_function = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5
    )

    epochs = 5

    print(
        "\nBeginning training..."
    )

    history = []

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for batch_number, batch in enumerate(
            train_loader,
            start=1
        ):

            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(device)

            graph_x = batch["graph_x"].to(device)
            edge_index = batch["edge_index"].to(device)

            labels = batch["labels"].unsqueeze(0).to(device)

            logits = model(
                input_ids,
                attention_mask,
                graph_x,
                edge_index
            )

            loss = loss_function(
                logits,
                labels
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if (
                batch_number % 50 == 0
            ):

                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"Batch {batch_number}/"
                    f"{len(train_loader)} "
                    f"Loss={loss.item():.4f}",
                    flush=True
                )

        average_loss = (
            total_loss /
            max(len(train_loader), 1)
        )

        val_macro, val_micro = evaluate(
            model,
            val_loader,
            device
        )

        print(
            f"\nEpoch {epoch + 1}/{epochs}"
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

        history.append({
            "epoch": epoch + 1,
            "loss": average_loss,
            "macro_f1": val_macro,
            "micro_f1": val_micro
        })

    # -------------------------
    # Test
    # -------------------------

    print(
        "\nEvaluating on test set..."
    )

    test_macro, test_micro = evaluate(
        model,
        test_loader,
        device
    )

    print(
        "\nFINAL FUSION TEST RESULTS"
    )

    print(
        "Macro-F1:",
        test_macro
    )

    print(
        "Micro-F1:",
        test_micro
    )

    # -------------------------
    # Save
    # -------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        MODELS_DIR /
        "fusion_model.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    results = {
        "macro_f1": test_macro,
        "micro_f1": test_micro,
        "history": history
    }

    results_path = (
        RESULTS_DIR /
        "fusion_results.json"
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
        "\nFusion model saved to:",
        model_path
    )

    print(
        "Fusion results saved to:",
        results_path
    )


if __name__ == "__main__":

    main()