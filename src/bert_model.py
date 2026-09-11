import torch
import torch.nn as nn

from transformers import AutoModel


class BERTTagClassifier(nn.Module):

    def __init__(
        self,
        model_name,
        num_labels
    ):

        super().__init__()

        self.bert = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Linear(
            hidden_size,
            num_labels
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # [CLS] representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(
            cls_embedding
        )

        return logits