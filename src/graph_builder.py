import numpy as np
import torch

from torch_geometric.data import Data

from config import GRAPH_SIMILARITY_THRESHOLD


def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return np.dot(a, b) / denominator


def build_graph(
    features,
    similarity_threshold=GRAPH_SIMILARITY_THRESHOLD
):

    num_nodes = len(features)

    edges = []

    # --------------------------------------------------
    # Temporal edges
    # --------------------------------------------------

    for i in range(num_nodes - 1):

        edges.append(
            [i, i + 1]
        )

        edges.append(
            [i + 1, i]
        )

    # --------------------------------------------------
    # Similarity edges
    # --------------------------------------------------

    for i in range(num_nodes):

        for j in range(i + 1, num_nodes):

            similarity = cosine_similarity(
                features[i],
                features[j]
            )

            if similarity >= similarity_threshold:

                edges.append(
                    [i, j]
                )

                edges.append(
                    [j, i]
                )

    if len(edges) == 0:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

    else:

        edge_index = torch.tensor(
            edges,
            dtype=torch.long
        ).t().contiguous()

    x = torch.tensor(
        features,
        dtype=torch.float32
    )

    graph = Data(
        x=x,
        edge_index=edge_index
    )

    return graph