import numpy as np
import os
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.optim as optim
import yaml

from helper_functions import create_electrical_grid_data, get_electrical_grid_data, ElectricalGridModel

# Get the current working directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Load configuration from YAML
with open(os.path.join(current_dir, "config.yaml"), "r") as config_file:
    config = yaml.safe_load(config_file)


# Use configuration parameters with full paths
injections_file = config["train_injections_file"]
adjacency_file = config["train_adjacency_file"]
loads_file = config["train_loads_file"]

data_train, data_val, data_test = get_electrical_grid_data(adjacency_file=adjacency_file,
                                                           injections_file=injections_file,
                                                           loads_file=loads_file)


# Initialize the model
model = ElectricalGridModel()

# Define hyperparameters
learning_rate = config["learning_rate"]
batch_size = config["batch_size"]
n_epochs = config["n_epochs"]

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Convert data splits to PyTorch DataLoader
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)

# Training loop
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Compute validation loss
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for data in val_loader:
            output = model(data)
            val_loss += criterion(output, data.y).item()

    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f'Epoch {epoch + 1}/{n_epochs}, Training Loss: {total_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}')

# Save the trained model checkpoint
torch.save(model.state_dict(), 'electrical_grid_model.pth')

# Calculate Bias
model.eval()
bias_list = []

with torch.no_grad():
    for data in test_loader:
        output = model(data)
        bias = torch.mean(output - data.y).item()
        bias_list.append(bias)

overall_bias = np.mean(bias_list)
print(f'Overall Bias: {overall_bias}')

# Analyze Bias Distribution
plt.hist(bias_list, bins=30, edgecolor='black')
plt.xlabel('Bias')
plt.ylabel('Frequency')
plt.title('Distribution of Bias')
plt.savefig('bias_distribution.png')
plt.show()