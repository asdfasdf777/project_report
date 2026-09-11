import ast
import pandas as pd
import torch

from torch.utils.data import Dataset


class FusionDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        graph_directory,
        max_length=128
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.tokenizer = tokenizer

        self.graph_directory = graph_directory

        self.max_length = max_length

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        track_id = int(
            row["track_id"]
        )

        # -------------------------
        # Text
        # -------------------------

        text = str(
            row["text"]
        )

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding[
            "input_ids"
        ].squeeze(0)

        attention_mask = encoding[
            "attention_mask"
        ].squeeze(0)

        # -------------------------
        # Graph
        # -------------------------

        graph_path = (
            self.graph_directory /
            f"{track_id:06d}.pt"
        )

        graph = torch.load(
            graph_path,
            weights_only=False
        )

        # -------------------------
        # Labels
        # -------------------------

        labels = torch.tensor(
            ast.literal_eval(
                row["labels"]
            ),
            dtype=torch.float32
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "graph_x": graph.x,
            "edge_index": graph.edge_index,
            "labels": labels,
            "track_id": track_id
        }