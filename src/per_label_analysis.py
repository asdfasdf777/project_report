import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    average_precision_score
)

from config import (
    RESULTS_DIR,
    SPLITS_DIR
)


MODEL_FILES = {
    "Majority": "majority_predictions.npz",
    "CNN": "cnn_predictions.npz",
    "GNN": "gnn_predictions.npz",
    "Context BERT": "context_bert_predictions.npz",
    "Fusion": "fusion_predictions.npz"
}


def load_labels():

    path = SPLITS_DIR / "labels.json"

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def analyze_model(
    model_name,
    filename,
    label_names
):

    path = (
        RESULTS_DIR
        / "predictions"
        / filename
    )

    data = np.load(path)

    probabilities = data["probabilities"]
    labels = data["labels"]

    predictions = (
        probabilities >= 0.5
    ).astype(np.float32)

    rows = []

    for i, label in enumerate(label_names):

        f1 = f1_score(
            labels[:, i],
            predictions[:, i],
            zero_division=0
        )

        auc_pr = average_precision_score(
            labels[:, i],
            probabilities[:, i]
        )

        prevalence = labels[:, i].mean()

        rows.append({
            "Model": model_name,
            "Label": label,
            "Prevalence": prevalence,
            "F1": f1,
            "AUC-PR": auc_pr
        })

    return rows


def main():

    label_names = load_labels()

    all_rows = []

    for model_name, filename in MODEL_FILES.items():

        rows = analyze_model(
            model_name,
            filename,
            label_names
        )

        all_rows.extend(rows)

    dataframe = pd.DataFrame(
        all_rows
    )

    output = (
        RESULTS_DIR
        / "per_label_results.csv"
    )

    dataframe.to_csv(
        output,
        index=False
    )

    print(
        "\nPer-label results saved to:",
        output
    )

    for model_name in MODEL_FILES:

        subset = dataframe[
            dataframe["Model"] == model_name
        ].sort_values(
            "F1",
            ascending=False
        )

        print(
            f"\n===== {model_name} ====="
        )

        print(
            subset[
                [
                    "Label",
                    "Prevalence",
                    "F1",
                    "AUC-PR"
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )


if __name__ == "__main__":
    main()