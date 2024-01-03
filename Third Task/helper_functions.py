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
