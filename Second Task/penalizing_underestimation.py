import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from helper_functions import get_electrical_grid_data, ElectricalGridModel

# Get the current working directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Load configuration from YAML
with open(os.path.join(current_dir, "config.yaml"), "r") as config_file:
    config = yaml.safe_load(config_file)


# Use configuration parameters with full paths
injections_file = config["train_injections_file"]
adjacency_file = config["train_adjacency_file"]
loads_file = config["train_loads_file"]

# Define a custom loss function with a penalty for underestimation
class CustomLoss(nn.Module):
    def __init__(self, penalty_weight=1.0):
        super(CustomLoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        mse = self.mse_loss(output, target)
        
        # Calculate the penalty for underestimation (only penalize negative errors)
        underestimation_penalty = torch.mean(torch.relu(output - target))

        # Combine MSE and underestimation penalty
        total_loss = mse + self.penalty_weight * underestimation_penalty
        return total_loss

# Load data and initialize the model
data_train, data_val, data_test = get_electrical_grid_data(adjacency_file, 
                                                           injections_file, 
                                                           loads_file)
model = ElectricalGridModel()

# Define hyperparameters
learning_rate = 0.01
batch_size = 64
n_epochs = 2000 

# Define loss function with penalty
penalty_weight = 0.1  # Adjust the penalty weight as needed
criterion = CustomLoss(penalty_weight)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Convert data splits to PyTorch DataLoader
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)

# Training loop with the custom loss
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

    print(f'Epoch {epoch + 1}/{n_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

# Save the trained model checkpoint
torch.save(model.state_dict(), 'electrical_grid_model_with_penalty.pth')

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
plt.savefig('bias_distribution_with_penalty.png')
# plt.show()