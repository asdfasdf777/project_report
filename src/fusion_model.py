import torch
import torch.nn as nn

from transformers import AutoModel

from torch_geometric.nn import SAGEConv


class FusionModel(nn.Module):

    def __init__(
        self,
        bert_name,
        num_labels,
        graph_input_dim=25,
        graph_hidden_dim=64
    ):

        super().__init__()

        # ---------------------------------------------
        # BERT
        # ---------------------------------------------

        self.bert = AutoModel.from_pretrained(
            bert_name
        )

        bert_dim = self.bert.config.hidden_size

        # ---------------------------------------------
        # GNN
        # ---------------------------------------------

        self.gnn1 = SAGEConv(
            graph_input_dim,
            graph_hidden_dim
        )

        self.gnn2 = SAGEConv(
            graph_hidden_dim,
            graph_hidden_dim
        )

        # ---------------------------------------------
        # Fusion classifier
        # ---------------------------------------------

        fusion_dim = (
            bert_dim +
            graph_hidden_dim
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_dim,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
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

        return outputs.last_hidden_state[:, 0, :]

    def encode_graph(
        self,
        x,
        edge_index
    ):

        x = self.gnn1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.gnn2(
            x,
            edge_index
        )

        x = torch.relu(x)

        return x.mean(
            dim=0,
            keepdim=True
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        graph_x,
        edge_index
    ):

        text_embedding = self.encode_text(
            input_ids,
            attention_mask
        )

        graph_embedding = self.encode_graph(
            graph_x,
            edge_index
        )

        combined = torch.cat(
            [
                text_embedding,
                graph_embedding
            ],
            dim=1
        )

        logits = self.classifier(
            combined
        )

        return logits