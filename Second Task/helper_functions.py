import numpy as np
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.optim as optim



def create_electrical_grid_data(node_features, target_values, adjacency_data):

    edge_index = []
    edge_attr = []

    for branch_id, branch_info in adjacency_data.items():
        from_node = branch_info['from']
        to_node = branch_info['to']
        reactance = branch_info['reactance']

        # Add edge indices
        edge_index.append([from_node, to_node])
        edge_index.append([to_node, from_node])  # Assuming an undirected graph

        # Add edge attributes
        edge_attr.append(reactance)
        edge_attr.append(reactance)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).view(-1, 1)

    # Create PyTorch Geometric Data object
    data = Data(x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=target_values)

    return data


def get_electrical_grid_data(adjacency_file, injections_file, loads_file):
    # Load edge connection and edge attribute information from 'adjacency.json'
    with open(adjacency_file, 'r') as f:
        adjacency_data = json.load(f)

    # Load input node information from 'injections.npy'
    node_features = np.load(injections_file)
    node_features = torch.tensor(node_features, dtype=torch.float32)

    # Load target values from 'loads.npy'
    target_values = np.load(loads_file)
    target_values = torch.tensor(target_values, dtype=torch.float32)

    # Constants
    TRAIN_SPLIT = 0.7  # 70% for training
    VAL_SPLIT = 0.15  # 15% for validation

    # Determine the sizes of training, validation, and test sets
    num_data_points = len(node_features)
    num_train = int(num_data_points * TRAIN_SPLIT)
    num_val = int(num_data_points * VAL_SPLIT)

    # Divide the dataset sequentially
    data_train = [create_electrical_grid_data(node_features[:num_train], target_values[:num_train], adjacency_data)]
    data_val = [create_electrical_grid_data(node_features[num_train:num_train + num_val], target_values[num_train:num_train + num_val], adjacency_data)]
    data_test = [create_electrical_grid_data(node_features[num_train + num_val:], target_values[num_train + num_val:], adjacency_data)]

    return data_train, data_val, data_test


# Define the ElectricalGridModel
class ElectricalGridModel(torch.nn.Module):
    def __init__(self, num_features=14, hidden_size=32, target_size=20, num_edge_features=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.num_edge_features = num_edge_features

        self.convs = [GATConv(self.num_features, self.hidden_size, edge_dim=self.num_edge_features),
                      GATConv(self.hidden_size, self.hidden_size, edge_dim=self.num_edge_features)]
        self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        x = self.linear(x)

        return x
