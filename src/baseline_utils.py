import numpy as np
import torch

from sklearn.metrics import (
    f1_score,
    average_precision_score
)


def calculate_metrics(
    probabilities,
    labels,
    threshold=0.5
):
    """
    Calculate multilabel Macro-F1, Micro-F1,
    Macro AUC-PR, and Micro AUC-PR.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    labels = np.asarray(
        labels,
        dtype=np.float32
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.float32)

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    micro_f1 = f1_score(
        labels,
        predictions,
        average="micro",
        zero_division=0
    )

    macro_auc_pr = average_precision_score(
        labels,
        probabilities,
        average="macro"
    )

    micro_auc_pr = average_precision_score(
        labels,
        probabilities,
        average="micro"
    )

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_auc_pr": float(macro_auc_pr),
        "micro_auc_pr": float(micro_auc_pr)
    }


def load_labels(dataframe):

    labels = []

    for value in dataframe["labels"]:

        if isinstance(value, str):
            import ast

            labels.append(
                ast.literal_eval(value)
            )

        else:
            labels.append(value)

    return np.asarray(
        labels,
        dtype=np.float32
    )