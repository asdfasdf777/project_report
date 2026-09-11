import ast
import json

import numpy as np
import pandas as pd

from config import (
    RESULTS_DIR,
    SPLITS_DIR
)


MODELS = {
    "Context BERT": "context_bert_predictions.npz",
    "GNN": "gnn_predictions.npz",
    "Fusion": "fusion_predictions.npz"
}


def load_labels():

    with open(
        SPLITS_DIR / "labels.json",
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def main():

    dataframe = pd.read_csv(
        SPLITS_DIR
        / "context_dataset.csv"
    )

    dataframe = dataframe[
        dataframe["split"] == "test"
    ].reset_index(drop=True)

    label_names = load_labels()

    results = []

    for model_name, filename in MODELS.items():

        path = (
            RESULTS_DIR
            / "predictions"
            / filename
        )

        data = np.load(path)

        probabilities = data[
            "probabilities"
        ]

        labels = data["labels"]

        for i in range(
            len(dataframe)
        ):

            true_labels = [
                label_names[j]
                for j in range(
                    len(label_names)
                )
                if labels[i, j] == 1
            ]

            top_indices = np.argsort(
                probabilities[i]
            )[::-1][:5]

            predictions = [
                (
                    label_names[j],
                    float(
                        probabilities[i, j]
                    )
                )
                for j in top_indices
            ]

            results.append({
                "Model": model_name,
                "Track ID": int(
                    dataframe.iloc[i][
                        "track_id"
                    ]
                ),
                "Text": dataframe.iloc[i][
                    "text"
                ],
                "Genre": dataframe.iloc[i][
                    "genre"
                ],
                "True Labels": ", ".join(
                    true_labels
                ),
                "Top Predictions": "; ".join(
                    [
                        f"{name} ({prob:.3f})"
                        for name, prob
                        in predictions
                    ]
                )
            })

    result_df = pd.DataFrame(
        results
    )

    output = (
        RESULTS_DIR
        / "case_studies.csv"
    )

    result_df.to_csv(
        output,
        index=False
    )

    print(
        "\nSaved:",
        output
    )

    # Print three BERT examples.
    bert_results = result_df[
        result_df["Model"] ==
        "Context BERT"
    ]

    print(
        "\n===== THREE BERT CASE STUDIES ====="
    )

    for i in range(
        min(3, len(bert_results))
    ):

        row = bert_results.iloc[i]

        print(
            f"\nCase Study {i + 1}"
        )

        print(
            "Track ID:",
            row["Track ID"]
        )

        print(
            "Context:",
            row["Text"]
        )

        print(
            "Genre:",
            row["Genre"]
        )

        print(
            "True labels:",
            row["True Labels"]
        )

        print(
            "Top predictions:",
            row["Top Predictions"]
        )


if __name__ == "__main__":
    main()