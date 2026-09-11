import json

import matplotlib.pyplot as plt
import pandas as pd

from config import RESULTS_DIR, PLOTS_DIR


MODELS = [
    ("Majority", "majority_results.json"),
    ("CNN", "cnn_results.json"),
    ("GNN", "gnn_results.json"),
    ("Context BERT", "context_bert_results.json"),
    ("Fusion", "fusion_results.json"),
]


def load_results():

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

    return pd.DataFrame(rows)


def make_plot(
    dataframe,
    metric,
    filename
):

    plt.figure(figsize=(10, 6))

    plt.bar(
        dataframe["Model"],
        dataframe[metric]
    )

    plt.ylabel(metric)
    plt.title(
        f"Model Comparison: {metric}"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    output = PLOTS_DIR / filename

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output)


def main():

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe = load_results()

    make_plot(
        dataframe,
        "Macro-F1",
        "macro_f1_comparison.png"
    )

    make_plot(
        dataframe,
        "Micro-F1",
        "micro_f1_comparison.png"
    )

    make_plot(
        dataframe,
        "Macro AUC-PR",
        "macro_auc_pr_comparison.png"
    )

    make_plot(
        dataframe,
        "Micro AUC-PR",
        "micro_auc_pr_comparison.png"
    )


if __name__ == "__main__":
    main()