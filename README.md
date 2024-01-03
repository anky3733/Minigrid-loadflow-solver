# Minigrid Loadflow Solver

## Depiction

This repository contains a simplified description and solver for an electricity grid. The grid consists of 14 busbars and 20 branches that connect these busbars. A busbar is the electrotechnical equivalent of a substation, and a branch is the equivalent of a powerline connecting substations.

## Dataset Description

The dataset comprises the following files:

- **adjacency.json**: Contains branch information. Each data point connects two busbars, specifying the reactance hindrance.
  Example data point: `"0": {"from": 0, "to": 1, "reactance": 0.05916999999999999}`.

- **injections.npy**: A (n_timesteps, n_bus) numpy array, where each timestep represents the power injected onto every bus.

- **loads.npy**: The target for prediction. A (n_timesteps, n_branch) numpy array representing the current on each branch.

## First Task

### 1. config.yaml

This configuration file requires users to provide paths to training and evaluation files. Additionally, configuration details such as learning rates, batch size, and the number of epochs can be specified.

### 2. helper_functions.py

Includes utility functions:
- `create_electrical_grid_data`: Converts injections, adjacency data, and loads data into a graph dataset.
- `get_electrical_grid_data`: Constructs a graph dataset from injections, adjacency data, and loads data.
- `ElectricalGridModel`: Defines a GATConv model for training.

### 3. train.py

This script is the main component of the first task:
- Loads data files based on the addresses provided in config files.
- Transforms the data into a Graph Dataset using `get_electrical_grid_data`.
- Trains the `ElectricalGridModel`.
- Validates performance using Mean Squared Error (MSE) Loss.
- Saves a model checkpoint.
- Plots Training Losses and Validation Losses with respect to the number of epochs.

### 4. eval.py

This script:
- Loads validation files from the address provided in the config files.
- Transforms the data into the proper format using `create_electrical_grid_data`.
- Loads the model saved during training.
- Makes predictions and saves them as `loads.npy`.
