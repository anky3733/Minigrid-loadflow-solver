import torch
from torch_geometric.data import Data
import numpy as np
import json
import os
import yaml

# Import your model and dataset functions
from helper_functions import create_electrical_grid_data, ElectricalGridModel

# Get the current working directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Load configuration from YAML
with open(os.path.join(current_dir, "config.yaml"), "r") as config_file:
    config = yaml.safe_load(config_file)


# Define file paths
injections_file_path = config['eval_injections_file']
adjacency_file_path = config['eval_adjacency_file']

# Load data for evaluation
node_features_eval = np.load(injections_file_path)
node_features_eval = torch.tensor(node_features_eval, dtype=torch.float32)

with open(adjacency_file_path, 'r') as f:
    adjacency_data_eval = json.load(f)

# Assuming your model has the same architecture and input dimensions
model = ElectricalGridModel()
model.load_state_dict(torch.load('electrical_grid_model.pth'))
model.eval()

data_eval = create_electrical_grid_data(node_features_eval, None, adjacency_data_eval)  # Set target_values to None

# Make predictions
with torch.no_grad():
    predictions = model(data_eval)

# Save the predictions as loads.npy
np.save('loads.npy', predictions)
