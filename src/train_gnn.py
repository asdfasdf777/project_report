import ast
import json
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from config import (
    PROCESSED_DIR,
    SPLITS_DIR,
    GNN_HIDDEN_DIM,
    NUM_LABELS,
    MODELS_DIR,
    RESULTS_DIR
)

from gnn_model import MusicGraphSAGE


class GraphDataset(Dataset):

    def __init__(self, dataframe):

        self.dataframe = dataframe.reset_index(drop=True)

        self.graph_directory = (
            PROCESSED_DIR / "graphs"
        )

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        track_id = int(row["track_id"])

        graph_path = (
            self.graph_directory /
            f"{track_id:06d}.pt"
        )

        graph = torch.load(
            graph_path,
            weights_only=False
        )

        labels = torch.tensor(
            ast.literal_eval(row["labels"]),
            dtype=torch.float32
        )

        graph.y = labels

        return graph


def calculate_f1(
    predictions,
    labels,
    threshold=0.5
):

    predictions = (
        torch.sigmoid(predictions) >= threshold
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

    macro_f1 = f1.mean()

    total_true_positive = true_positive.sum()
    total_false_positive = false_positive.sum()
    total_false_negative = false_negative.sum()

    micro_precision = (
        total_true_positive /
        (
            total_true_positive +
            total_false_positive +
            1e-8
        )
    )

    micro_recall = (
        total_true_positive /
        (
            total_true_positive +
            total_false_negative +
            1e-8
        )
    )

    micro_f1 = (
        2 * micro_precision * micro_recall /
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

        for graph in loader:

            graph = graph.to(device)

            logits = model(
                graph.x,
                graph.edge_index
            )

            all_predictions.append(
                logits.cpu()
            )

            all_labels.append(
                graph.y.cpu()
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

    # PyTorch Geometric concatenates 1D graph labels.
    # Restore one 20-label vector per graph.
    labels = labels.view(
        -1,
        NUM_LABELS
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

    torch.manual_seed(42)

    if torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print(
        "Device:",
        device
    )

    dataset_path = (
        SPLITS_DIR / "dataset.csv"
    )

    dataframe = pd.read_csv(
        dataset_path
    )

    print(
        "Dataset rows:",
        len(dataframe)
    )

    print(
        "Dataset split counts:"
    )

    print(
        dataframe["split"].value_counts()
    )

    graph_directory = (
        PROCESSED_DIR / "graphs"
    )

    available_rows = []

    for _, row in dataframe.iterrows():

        track_id = int(row["track_id"])

        graph_path = (
            graph_directory /
            f"{track_id:06d}.pt"
        )

        if graph_path.exists():

            available_rows.append(row)

    dataframe = pd.DataFrame(
        available_rows
    )

    print(
        "\nRows with graph files:",
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
        "Train graphs:",
        len(train_df)
    )

    print(
        "Validation graphs:",
        len(val_df)
    )

    print(
        "Test graphs:",
        len(test_df)
    )

    train_dataset = GraphDataset(
        train_df
    )

    val_dataset = GraphDataset(
        val_df
    )

    test_dataset = GraphDataset(
        test_df
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    print(
        "\nCreating GNN..."
    )

    model = MusicGraphSAGE(
        input_dim=25,
        hidden_dim=GNN_HIDDEN_DIM,
        num_labels=NUM_LABELS
    )

    model = model.to(device)

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    print(
        "Beginning training..."
    )

    for epoch in range(5):

        model.train()

        total_loss = 0.0

        for graph in train_loader:

            graph = graph.to(device)

            logits = model(
                graph.x,
                graph.edge_index
            )

            labels = graph.y

            if labels.dim() == 1:

                labels = labels.unsqueeze(0)

            loss = loss_function(
                logits,
                labels
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

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
            f"Epoch {epoch + 1}/5 "
            f"Loss={average_loss:.4f} "
            f"Macro-F1={val_macro:.4f} "
            f"Micro-F1={val_micro:.4f}"
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
        "\nFINAL GNN TEST RESULTS"
    )

    print(
        "Macro-F1:",
        test_macro
    )

    print(
        "Micro-F1:",
        test_micro
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
        MODELS_DIR / "gnn_model.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    results = {
        "macro_f1": test_macro,
        "micro_f1": test_micro
    }

    results_path = (
        RESULTS_DIR / "gnn_results.json"
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
        "\nModel saved to:",
        model_path
    )

    print(
        "Results saved to:",
        results_path
    )


if __name__ == "__main__":

    main()