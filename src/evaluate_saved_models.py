import ast
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from config import (
    SPLITS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    BERT_MODEL_NAME,
    MAX_TEXT_LENGTH,
    NUM_LABELS,
    GNN_HIDDEN_DIM,
    MODELS_DIR
)

from bert_model import BERTTagClassifier
from gnn_model import MusicGraphSAGE
from fusion_model import FusionModel
from baseline_utils import calculate_metrics


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_dataframe():
    path = SPLITS_DIR / "context_dataset.csv"
    dataframe = pd.read_csv(path)

    return (
        dataframe[dataframe["split"] == "test"].reset_index(drop=True)
    )


def load_labels(dataframe):
    labels = []

    for value in dataframe["labels"]:
        labels.append(
            ast.literal_eval(value)
        )

    return np.asarray(
        labels,
        dtype=np.float32
    )


def evaluate_bert(test_df):
    print("\nEvaluating Context BERT...")

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_NAME
    )

    model = BERTTagClassifier(
        BERT_MODEL_NAME,
        NUM_LABELS
    )

    model_path = (
        MODELS_DIR / "context_bert_model.pt"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.to(DEVICE)
    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for _, row in test_df.iterrows():

            encoding = tokenizer(
                str(row["text"]),
                padding="max_length",
                truncation=True,
                max_length=MAX_TEXT_LENGTH,
                return_tensors="pt"
            )

            input_ids = (
                encoding["input_ids"]
                .to(DEVICE)
            )

            attention_mask = (
                encoding["attention_mask"]
                .to(DEVICE)
            )

            logits = model(
                input_ids,
                attention_mask
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.append(
                probs.cpu().numpy()[0]
            )

            labels.append(
                ast.literal_eval(
                    row["labels"]
                )
            )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    labels = np.asarray(
        labels,
        dtype=np.float32
    )

    metrics = calculate_metrics(
        probabilities,
        labels
    )

    save_results(
        "context_bert",
        metrics,
        probabilities,
        labels,
        test_df
    )

    print_results(
        "CONTEXT BERT",
        metrics
    )


def evaluate_gnn(test_df):
    print("\nEvaluating GNN...")

    model = MusicGraphSAGE(
        input_dim=25,
        hidden_dim=GNN_HIDDEN_DIM,
        num_labels=NUM_LABELS
    )

    model_path = (
        MODELS_DIR / "gnn_model.pt"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.to(DEVICE)
    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for _, row in test_df.iterrows():

            track_id = int(
                row["track_id"]
            )

            graph_path = (
                PROCESSED_DIR
                / "graphs"
                / f"{track_id:06d}.pt"
            )

            graph = torch.load(
                graph_path,
                map_location=DEVICE,
                weights_only=False
            )

            logits = model(
                graph.x.to(DEVICE),
                graph.edge_index.to(DEVICE)
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.append(
                probs.cpu().numpy()[0]
            )

            labels.append(
                ast.literal_eval(
                    row["labels"]
                )
            )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    labels = np.asarray(
        labels,
        dtype=np.float32
    )

    metrics = calculate_metrics(
        probabilities,
        labels
    )

    save_results(
        "gnn",
        metrics,
        probabilities,
        labels,
        test_df
    )

    print_results(
        "GNN",
        metrics
    )


def evaluate_fusion(test_df):
    print("\nEvaluating Fusion Model...")

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_NAME
    )

    model = FusionModel(
        BERT_MODEL_NAME,
        NUM_LABELS,
        graph_input_dim=25,
        graph_hidden_dim=GNN_HIDDEN_DIM
    )

    model_path = (
        MODELS_DIR / "fusion_model.pt"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.to(DEVICE)
    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for _, row in test_df.iterrows():

            track_id = int(
                row["track_id"]
            )

            encoding = tokenizer(
                str(row["text"]),
                padding="max_length",
                truncation=True,
                max_length=MAX_TEXT_LENGTH,
                return_tensors="pt"
            )

            input_ids = (
                encoding["input_ids"]
                .to(DEVICE)
            )

            attention_mask = (
                encoding["attention_mask"]
                .to(DEVICE)
            )

            graph_path = (
                PROCESSED_DIR
                / "graphs"
                / f"{track_id:06d}.pt"
            )

            graph = torch.load(
                graph_path,
                map_location=DEVICE,
                weights_only=False
            )

            logits = model(
                input_ids,
                attention_mask,
                graph.x.to(DEVICE),
                graph.edge_index.to(DEVICE)
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.append(
                probs.cpu().numpy()[0]
            )

            labels.append(
                ast.literal_eval(
                    row["labels"]
                )
            )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    labels = np.asarray(
        labels,
        dtype=np.float32
    )

    metrics = calculate_metrics(
        probabilities,
        labels
    )

    save_results(
        "fusion",
        metrics,
        probabilities,
        labels,
        test_df
    )

    print_results(
        "FUSION",
        metrics
    )


def save_results(
    name,
    metrics,
    probabilities,
    labels,
    dataframe
):
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

    prediction_path = (
        predictions_dir
        / f"{name}_predictions.npz"
    )

    np.savez(
        prediction_path,
        probabilities=probabilities,
        labels=labels,
        track_ids=dataframe[
            "track_id"
        ].to_numpy()
    )

    results = {
        "model": name,
        "num_labels": NUM_LABELS,
        "metrics": metrics
    }

    results_path = (
        RESULTS_DIR
        / f"{name}_results.json"
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
        "Results saved to:",
        results_path
    )

    print(
        "Predictions saved to:",
        prediction_path
    )


def print_results(
    name,
    metrics
):
    print(
        f"\n{name} TEST RESULTS"
    )

    print(
        f"Macro-F1: "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Micro-F1: "
        f"{metrics['micro_f1']:.4f}"
    )

    print(
        f"Macro AUC-PR: "
        f"{metrics['macro_auc_pr']:.4f}"
    )

    print(
        f"Micro AUC-PR: "
        f"{metrics['micro_auc_pr']:.4f}"
    )


def main():

    print(
        "Device:",
        DEVICE
    )

    test_df = load_dataframe()

    print(
        "Test samples:",
        len(test_df)
    )

    evaluate_bert(test_df)
    evaluate_gnn(test_df)
    evaluate_fusion(test_df)

    print(
        "\nAll saved-model evaluations completed."
    )


if __name__ == "__main__":
    main()