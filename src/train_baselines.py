import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_utils import (
    calculate_metrics,
    load_labels
)

from config import (
    SPLITS_DIR,
    RESULTS_DIR,
    NUM_LABELS
)


def main():

    print(
        "Starting baseline experiments..."
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

    train_labels = load_labels(
        train_df
    )

    val_labels = load_labels(
        val_df
    )

    test_labels = load_labels(
        test_df
    )

    print(
        "Training samples:",
        len(train_df)
    )

    print(
        "Validation samples:",
        len(val_df)
    )

    print(
        "Test samples:",
        len(test_df)
    )

    # --------------------------------------------------
    # MAJORITY BASELINE
    # --------------------------------------------------

    print(
        "\nCalculating majority baseline..."
    )

    label_prevalence = (
        train_labels.mean(axis=0)
    )

    majority_predictions = (
        label_prevalence >= 0.5
    ).astype(np.float32)

    majority_probabilities = np.tile(
        label_prevalence,
        (
            len(test_labels),
            1
        )
    )

    majority_metrics = calculate_metrics(
        majority_probabilities,
        test_labels
    )

    print(
        "\nMAJORITY BASELINE TEST RESULTS"
    )

    print(
        f"Macro-F1: "
        f"{majority_metrics['macro_f1']:.4f}"
    )

    print(
        f"Micro-F1: "
        f"{majority_metrics['micro_f1']:.4f}"
    )

    print(
        f"Macro AUC-PR: "
        f"{majority_metrics['macro_auc_pr']:.4f}"
    )

    print(
        f"Micro AUC-PR: "
        f"{majority_metrics['micro_auc_pr']:.4f}"
    )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_dir = (
        RESULTS_DIR
        / "predictions"
    )

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    majority_prediction_file = (
        predictions_dir
        / "majority_predictions.npz"
    )

    np.savez(
        majority_prediction_file,
        probabilities=majority_probabilities,
        labels=test_labels,
        track_ids=test_df[
            "track_id"
        ].to_numpy()
    )

    results = {
        "model": "majority",
        "num_labels": NUM_LABELS,
        "label_prevalence": (
            label_prevalence.tolist()
        ),
        "metrics": majority_metrics
    }

    results_file = (
        RESULTS_DIR
        / "majority_results.json"
    )

    with open(
        results_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        "\nMajority results saved to:",
        results_file
    )

    print(
        "Majority predictions saved to:",
        majority_prediction_file
    )


if __name__ == "__main__":
    main()