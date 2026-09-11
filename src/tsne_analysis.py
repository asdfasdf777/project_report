import ast
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.manifold import TSNE
from transformers import AutoTokenizer

from config import (
    SPLITS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    MODELS_DIR,
    BERT_MODEL_NAME,
    MAX_TEXT_LENGTH,
    NUM_LABELS,
    GNN_HIDDEN_DIM
)

from bert_model import BERTTagClassifier
from gnn_model import MusicGraphSAGE
from fusion_model import FusionModel


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def load_test_data():

    dataframe = pd.read_csv(
        SPLITS_DIR
        / "context_dataset.csv"
    )

    return dataframe[
        dataframe["split"] == "test"
    ].reset_index(drop=True)


def load_bert():

    model = BERTTagClassifier(
        BERT_MODEL_NAME,
        NUM_LABELS
    )

    model.load_state_dict(
        torch.load(
            MODELS_DIR / "context_bert_model.pt",
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


def load_gnn():

    model = MusicGraphSAGE(
        input_dim=25,
        hidden_dim=GNN_HIDDEN_DIM,
        num_labels=NUM_LABELS
    )

    model.load_state_dict(
        torch.load(
            MODELS_DIR / "gnn_model.pt",
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


def load_fusion():

    model = FusionModel(
        BERT_MODEL_NAME,
        NUM_LABELS,
        graph_input_dim=25,
        graph_hidden_dim=GNN_HIDDEN_DIM
    )

    model.load_state_dict(
        torch.load(
            MODELS_DIR / "fusion_model.pt",
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


def collect_embeddings(
    dataframe
):

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_NAME
    )

    bert = load_bert()
    gnn = load_gnn()
    fusion = load_fusion()

    bert_embeddings = []
    gnn_embeddings = []
    fusion_embeddings = []
    labels = []

    with torch.no_grad():

        for _, row in dataframe.iterrows():

            track_id = int(
                row["track_id"]
            )

            encoding = tokenizer(
                str(row["text"]),
                padding="max_length",
                truncation=True,
                max_length=MAX_TEXT_LENGTH,
                return_tensors="pt"
            )

            input_ids = (
                encoding["input_ids"]
                .to(DEVICE)
            )

            attention_mask = (
                encoding["attention_mask"]
                .to(DEVICE)
            )

            bert_output = bert.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            bert_embedding = (
                bert_output
                .last_hidden_state[:, 0, :]
                .cpu()
                .numpy()[0]
            )

            graph_path = (
                PROCESSED_DIR
                / "graphs"
                / f"{track_id:06d}.pt"
            )

            graph = torch.load(
                graph_path,
                map_location=DEVICE,
                weights_only=False
            )

            graph_x = graph.x.to(DEVICE)
            edge_index = graph.edge_index.to(
                DEVICE
            )

            graph_embedding = (
                gnn.encode_graph(
                    graph_x,
                    edge_index
                )
                .cpu()
                .numpy()[0]
            )

            fusion_text = fusion.encode_text(
                input_ids,
                attention_mask
            )

            fusion_graph = fusion.encode_graph(
                graph_x,
                edge_index
            )

            fusion_embedding = torch.cat(
                [
                    fusion_text,
                    fusion_graph
                ],
                dim=1
            ).cpu().numpy()[0]

            bert_embeddings.append(
                bert_embedding
            )

            gnn_embeddings.append(
                graph_embedding
            )

            fusion_embeddings.append(
                fusion_embedding
            )

            label_vector = np.asarray(
                ast.literal_eval(
                    row["labels"]
                )
            )

            labels.append(
                label_vector
            )

    return (
        np.asarray(bert_embeddings),
        np.asarray(gnn_embeddings),
        np.asarray(fusion_embeddings),
        np.asarray(labels)
    )


def make_tsne(
    embeddings,
    labels,
    title,
    filename
):

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30
    )

    reduced = tsne.fit_transform(
        embeddings
    )

    # Use the number of active labels as
    # a simple color value.
    label_count = labels.sum(axis=1)

    plt.figure(figsize=(9, 7))

    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=label_count,
        alpha=0.8
    )

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(title)

    colorbar = plt.colorbar()
    colorbar.set_label(
        "Number of active labels"
    )

    plt.tight_layout()

    output = PLOTS_DIR / filename

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output
    )


def main():

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe = load_test_data()

    (
        bert,
        gnn,
        fusion,
        labels
    ) = collect_embeddings(
        dataframe
    )

    np.savez(
        RESULTS_DIR
        / "tsne_embeddings.npz",
        bert=bert,
        gnn=gnn,
        fusion=fusion,
        labels=labels,
        track_ids=dataframe[
            "track_id"
        ].to_numpy()
    )

    make_tsne(
        bert,
        labels,
        "t-SNE of Context BERT Embeddings",
        "tsne_bert.png"
    )

    make_tsne(
        gnn,
        labels,
        "t-SNE of GNN Embeddings",
        "tsne_gnn.png"
    )

    make_tsne(
        fusion,
        labels,
        "t-SNE of Early-Fusion Embeddings",
        "tsne_fusion.png"
    )


if __name__ == "__main__":
    main()