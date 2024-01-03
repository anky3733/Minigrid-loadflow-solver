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

## Second Task

### Documented Thoughts to potentially reduce underestimation at the cost of more overestimation.

### Measuring Bias

**1. Calculate Bias:**
Compute the bias of your model by comparing the predicted values to the actual values. The bias can be calculated as the mean difference between predicted and actual values. Positive bias indicates overestimation, while negative bias indicates underestimation.

**2. Analyze Bias Distribution:**
Examine the distribution of bias values to understand the magnitude and direction of the bias across different branches or nodes in the electrical grid.

### Strategies to Address Underestimation

**Penalizing Underestimation:**
Modify the loss function to include a penalty term for underestimation. This penalty term could be a function of the absolute value of the difference between predicted and actual values.

**bias_estimation.py:**
In this script, we use the model trained in the first task. We calculate the bias as mentioned in the section - Calculating Bias and analyze the distribution. We achieved an overall bias score of -0.1411.

**penalizing_underestimation.py:**
In this script, we modify the loss function to include a penalty term for cases where the model underestimates the target values. We achieved an overall bias score of -0.3120.

The higher negative bias suggests that the penalty has successfully influenced the model to be more cautious about underestimation. This implies that the model, with the penalty, is now more inclined to predict values that are closer to or higher than the actual loads. The penalty for underestimation has shifted the model's behavior towards reducing underestimation, even if it comes at the cost of potentially having more overestimation.
