import torch
import torch.nn as nn

from transformers import AutoModel
from torch_geometric.nn import SAGEConv


class CrossAttentionFusionModel(
    nn.Module
):

    def __init__(
        self,
        bert_name,
        num_labels,
        graph_input_dim=25,
        graph_hidden_dim=64,
        attention_heads=8
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(
            bert_name
        )

        bert_dim = (
            self.bert.config.hidden_size
        )

        self.gnn1 = SAGEConv(
            graph_input_dim,
            graph_hidden_dim
        )

        self.gnn2 = SAGEConv(
            graph_hidden_dim,
            graph_hidden_dim
        )

        self.graph_projection = nn.Linear(
            graph_hidden_dim,
            bert_dim
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=bert_dim,
            num_heads=attention_heads,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                bert_dim,
                256
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                256,
                num_labels
            )
        )

    def encode_text(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return outputs.last_hidden_state

    def encode_graph(
        self,
        graph_x,
        edge_index
    ):

        x = self.gnn1(
            graph_x,
            edge_index
        )

        x = torch.relu(x)

        x = self.gnn2(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.graph_projection(x)

        return x

    def forward(
        self,
        input_ids,
        attention_mask,
        graph_x,
        edge_index
    ):

        text_tokens = self.encode_text(
            input_ids,
            attention_mask
        )

        graph_nodes = self.encode_graph(
            graph_x,
            edge_index
        )

        graph_nodes = graph_nodes.unsqueeze(0)

        attended_text, _ = self.cross_attention(
            query=graph_nodes,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=(
                attention_mask == 0
            )
        )

        fused_embedding = (
            attended_text.mean(
                dim=1
            )
        )

        logits = self.classifier(
            fused_embedding
        )

        return logits