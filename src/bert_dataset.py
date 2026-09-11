import ast

import pandas as pd
import torch

from torch.utils.data import Dataset


class MusicTextDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=128
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.tokenizer = tokenizer

        self.max_length = max_length

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        text = str(row["text"])

        labels = torch.tensor(
            ast.literal_eval(row["labels"]),
            dtype=torch.float32
        )

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels
        }

        return item