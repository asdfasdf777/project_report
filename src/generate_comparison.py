import json

import pandas as pd

from config import RESULTS_DIR


MODELS = [
    ("Majority", "majority_results.json"),
    ("CNN", "cnn_results.json"),
    ("GNN", "gnn_results.json"),
    ("Context BERT", "context_bert_results.json"),
    ("Early-Concatenation Fusion", "fusion_results.json"),
    ("Cross-Attention Fusion", "cross_attention_results.json"),
]


def main():

    rows = []

    for name, filename in MODELS:

        path = RESULTS_DIR / filename

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        metrics = data["metrics"]

        rows.append({
            "Model": name,
            "Macro-F1": metrics["macro_f1"],
            "Micro-F1": metrics["micro_f1"],
            "Macro AUC-PR": metrics["macro_auc_pr"],
            "Micro AUC-PR": metrics["micro_auc_pr"]
        })

    dataframe = pd.DataFrame(rows)

    output = (
        RESULTS_DIR
        / "final_comparison.csv"
    )

    dataframe.to_csv(
        output,
        index=False
    )

    print(
        dataframe.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print(
        "\nSaved:",
        output
    )


if __name__ == "__main__":
    main()