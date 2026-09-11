import json

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

from config import (
    RESULTS_DIR,
    SPLITS_DIR
)


MODEL_FILES = {
    "Majority": "majority_predictions.npz",
    "CNN": "cnn_predictions.npz",
    "GNN": "gnn_predictions.npz",
    "Context BERT": "context_bert_predictions.npz",
    "Fusion": "fusion_predictions.npz",
    "Cross-Attention": "cross_attention_predictions.npz"
}


def calculate_best_threshold(
    probabilities,
    labels
):

    thresholds = np.arange(
        0.05,
        0.96,
        0.05
    )

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(np.float32)

        score = f1_score(
            labels,
            predictions,
            average="micro",
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    return (
        best_threshold,
        best_f1
    )


def main():

    print(
        "Threshold analysis is based on "
        "the saved test predictions."
    )

    rows = []

    for model_name, filename in MODEL_FILES.items():

        path = (
            RESULTS_DIR
            / "predictions"
            / filename
        )

        if not path.exists():

            print(
                "Skipping missing:",
                filename
            )

            continue

        data = np.load(path)

        probabilities = data[
            "probabilities"
        ]

        labels = data["labels"]

        threshold, f1 = (
            calculate_best_threshold(
                probabilities,
                labels
            )
        )

        rows.append({
            "Model": model_name,
            "Best Threshold": threshold,
            "Micro-F1 at Threshold": f1
        })

    dataframe = pd.DataFrame(
        rows
    )

    output = (
        RESULTS_DIR
        / "threshold_analysis.csv"
    )

    dataframe.to_csv(
        output,
        index=False
    )

    print(
        dataframe.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        output
    )


if __name__ == "__main__":
    main()