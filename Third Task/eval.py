import numpy as np
import json
import os
import yaml

import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch
from torch_geometric.data import Data

# Import your model and dataset functions
from helper_functions import create_electrical_grid_data, get_electrical_grid_data, ElectricalGridModel

# Get the current working directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Load configuration from YAML
with open(os.path.join(current_dir, "config.yaml"), "r") as config_file:
    config = yaml.safe_load(config_file)

# Define file paths
injections_file = config['eval_injections_file']
adjacency_file = config['eval_adjacency_file']

# Load data for evaluation
node_features_eval = np.load(injections_file)
node_features_eval = torch.tensor(node_features_eval, dtype=torch.float32)


with open(adjacency_file, 'r') as f:
    adjacency_data_eval = json.load(f)

data = create_electrical_grid_data(node_features_eval, None, adjacency_data_eval)


class ExtendedElectricalGridModel(torch.nn.Module):
    def __init__(self, num_features=15, hidden_size=32, target_size=20, num_edge_features=1):
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

# Load the extended model
extended_model = ExtendedElectricalGridModel()

# Load the weights from the pre-trained model
extended_model.load_state_dict(torch.load('electrical_grid_model.pth'))

# Set the model to evaluation mode
extended_model.eval()

# Assuming data_eval is your validation data with shape torch.Size([50, 15])
with torch.no_grad():
    predictions = extended_model(data)
    # Process predictions as needed


# Save the predictions as loads.npy
np.save('loads.npy', predictions)
