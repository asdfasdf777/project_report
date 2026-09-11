import torch
import torch.nn as nn

from torch_geometric.nn import SAGEConv


class MusicGraphSAGE(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_labels
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        self.relu = nn.ReLU()

        self.classifier = nn.Linear(
            hidden_dim,
            num_labels
        )

    def encode_graph(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = self.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = self.relu(x)

        # Mean pooling
        graph_embedding = x.mean(
            dim=0,
            keepdim=True
        )

        return graph_embedding

    def forward(
        self,
        x,
        edge_index
    ):

        graph_embedding = self.encode_graph(
            x,
            edge_index
        )

        logits = self.classifier(
            graph_embedding
        )

        return logits