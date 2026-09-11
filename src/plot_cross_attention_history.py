import json

import matplotlib.pyplot as plt

from config import RESULTS_DIR, PLOTS_DIR


def main():

    with open(
        RESULTS_DIR
        / "cross_attention_results.json",
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    history = data["history"]

    epochs = [
        item["epoch"]
        for item in history
    ]

    losses = [
        item["loss"]
        for item in history
    ]

    macro_f1 = [
        item["macro_f1"]
        for item in history
    ]

    micro_f1 = [
        item["micro_f1"]
        for item in history
    ]

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        losses,
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(
        "Cross-Attention Training Loss"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "cross_attention_loss.png",
        dpi=300
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        macro_f1,
        marker="o",
        label="Macro-F1"
    )

    plt.plot(
        epochs,
        micro_f1,
        marker="o",
        label="Micro-F1"
    )

    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title(
        "Cross-Attention Validation F1"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "cross_attention_validation_f1.png",
        dpi=300
    )

    plt.close()

    print(
        "Cross-attention plots generated."
    )


if __name__ == "__main__":
    main()