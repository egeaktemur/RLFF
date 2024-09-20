# reported by stackoverflow due to module error (downgrading matplotlib also possible)
import matplotlib
matplotlib.use('TkAgg')

import os
from itertools import combinations
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
import gymnasium as gym
from networks import FFNet
from helper import overlay_y_on_x, move
import logging


DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
print(DEVICE)

ENV_NAME = "LunarLander-v2"
HIDDEN_DIMS = [2000, 2000, 2000, 2000]  # Set hidden layers as 3x1000
N_TRAJ_SAMPLES = 5
MAX_STEPS = 500
EPOCHS = 3000
# For LunarLander the range of 8-dimensional state space is: [(-1.5, 1.5), (-1.5, 1.5), (-5, 5), (-5, 5), (-3.14, 3.14), (-5, 5), {0, 1}, {0, 1}]
STATE_DIM_THRESHOLDS = [0.03, 0.03, 0.1, 0.1, 0.0628, 0.1, 1, 1]
GAMMA = 0.99

LOG_STEPS = 50
SAVE_STEPS = 500

###################
# Data Generation #
###################


def sample_trajectories(env, policy, n, max_steps=np.inf, gamma=0.99):
    """
    Sample n trajectories using the defined strategy and the policy.

    Parameters:
    - env: Gym environment
    - policy: A function that takes a state and returns an action probability distribution
    - n: Number of trajectories to sample
    - max_steps: Maximum number of steps per trajectory (might lead to underestimated returns, because we do not consider future rewards like in Q-learning)
    - gamma: Discount factor

    Returns:
    - trajectories: A list of trajectories, where each trajectory is a list of (state, action, reward, next_state, done) tuples
    """
    trajectories = []
    total_reward = 0
    for i in range(n):
        state, _ = env.reset()
        done = False
        trajectory = []

        while True:
            action_probs = policy(state)
            action = np.random.choice(len(action_probs), p=action_probs)

            next_state, reward, done, _, _ = env.step(action)
            trajectory.append([state, action, reward, 0])
            state = next_state
            total_reward += reward

            if done or len(trajectory) >= max_steps:
                break

        # calculate discounted returns
        running_total = 0
        for i in reversed(range(len(trajectory))):
            running_total = running_total * gamma + trajectory[i][2]
            trajectory[i][3] = running_total

        trajectories.append(trajectory)

    return trajectories, total_reward / n


# one strategy to estimate long-term rewards
def bin_states(trajectories, thresholds, plot=False):
    """
    Bin states across all trajectories based on a threshold per state dimension.

    Parameters:
    - trajectories: List of trajectories, where each trajectory is a list of (state, action, reward, next_state, long-term reward) tuples
    - thresholds: List of threshold values for each state dimension
    - plot: Whether to plot the bin distribution

    Returns:
    - bins: A dictionary where each key is a bin (tuple representing bin index) and the value is a list of states in that bin
    """
    bins = {}

    for trajectory in trajectories:
        for state, action, reward, lt_reward in trajectory:
            # perform rounded division by threshold to get bin index
            bin_index = tuple((state // thresholds).astype(int))

            if bin_index not in bins:
                bins[bin_index] = []

            bins[bin_index].append((state, action, lt_reward))

    if plot:
        # get an overview of the number of states in each bin
        nb_states = np.array([len(v) for k, v in bins.items()])
        unique, counts = np.unique(nb_states, return_counts=True)
        # plot the bin distribution
        plot_bar(unique, counts, "Bin Distribution", "Number of States", "Number of Bins")
        print(f"Number of Bins (different States): {len(bins)}")

    return bins


def plot_bar(keys, counts, title, xlabel, ylabel):
    plt.bar(keys, counts)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig("bin_dist.png")
    # plt.show()

##############################
# Training + FFNet Functions #
##############################

# => adjusted for RL setting in order to not change networks.py


def get_action_probs(state, ffnet, method="Goodness"):
    """
    Get the action probabilities for a given state using the FFNet and a defined method.
    """
    # adjust the input to include action dimensions (and neutral input in case of classifier)
    state_ = torch.FloatTensor(state[np.newaxis, :]).cuda()
    ff_input = overlay_y_on_x(state_, neu=True, output_dim=ffnet.output_dim, add=True)
    with torch.no_grad():
        if method == "Classifier":
            logits = ffnet.classifier(ffnet.get_activations(ff_input))
        if method == "Goodness":
            logits = ffnet(ff_input)
    probabilities = ffnet.softmax(logits)
    return probabilities.cpu().numpy()[0]


def train_ff_model(model, env, epochs,  n, max_steps, gamma, thresholds, method="Goodness", save_path="ff_model"):
    """
    Train a neural network via forward-forward in an RL setting.

    Parameters:
    - model: The FF Network to train
    - env: Gym environment
    - epochs: Number of training epochs
    - n: Number of trajectories to sample
    - max_steps: Maximum number of steps per trajectory
    - gamma: Discount factor
    - thresholds: List of threshold values for each state dimension to match states
    - method: Method for FF Training (Goodness | Classifier)
    - save_path: Path to save the model
    """

    model.train()  # Set the model to training mode
    model.epochs = epochs
    print(f"Epoch 1/{epochs}")
    for epoch in range(1, epochs + 1):
        if epoch % LOG_STEPS == 0:
            print(f"Epoch {epoch}/{epochs}")
        # wrap current model into a policy (input a state and output action probabilities
        current_policy = partial(get_action_probs, ffnet=model, method=method)
        # sample trajectories
        trajectories, avg_total_reward = sample_trajectories(env, current_policy, n, max_steps, gamma)
        if epoch % LOG_STEPS == 0:
            print(f"Average Total Reward per Episode: {avg_total_reward}")
            logging.info('', extra={'epoch': epoch, 'avg_total_reward': avg_total_reward})

        # bin states
        bins = bin_states(trajectories, thresholds, plot=False)
        pos_states = []
        pos_actions = []
        neg_states = []
        neg_actions = []

        # create positive and negative data by iterating over bins and creating pairs
        for key, value in bins.items():
            # d1 and d2 are tuples of (state, action, lt_reward)
            for d1, d2 in combinations(value, 2):
                # NOTE: here currently just for the Goodness method Todo: add for Classifier
                # continue if the actions are the same
                if d1[1] == d2[1]:
                    continue
                if d1[2] > d2[2]:
                    pos_states.append(d1[0])
                    pos_actions.append(d1[1])
                    neg_states.append(d2[0])
                    neg_actions.append(d2[1])
                elif d1[2] < d2[2]:
                    pos_states.append(d2[0])
                    pos_actions.append(d2[1])
                    neg_states.append(d1[0])
                    neg_actions.append(d1[1])
        if epoch % LOG_STEPS == 0:
            print(f"Number of positive/negative data points: {len(pos_states)}")
        if len(pos_states) == 0:
            print(f"No positive/negative data points found. Skipping epoch {epoch}.")
            continue
        # construct training input with overlay
        pos_data = overlay_y_on_x(move(np.array(pos_states)), pos_actions, neu=False, output_dim=model.output_dim, add=True)
        neg_data = overlay_y_on_x(move(np.array(neg_states)), neg_actions, neu=False, output_dim=model.output_dim, add=True)

        neu_data = overlay_y_on_x(move(np.array(pos_states)), neu=True, output_dim=model.output_dim, add=True)
        labels = pos_actions

        model.total_size = len(pos_data)
        for layer_index, layer in enumerate(model.layers):
            layer_loss = 0
            model.learning_rate_cooldown(epoch+1)
            #layer_loss += layer.train_layer(pos_data, neg_data)
            for start_index, end_index in model.generate_batches():
                layer_loss += layer.train_layer(
                    pos_data[start_index:end_index], neg_data[start_index:end_index])
            if layer_index != len(model.layers) - 1:
                pos_data, neg_data = layer.output(pos_data), layer.output(neg_data)
            if epoch % LOG_STEPS == 0:
                print(f"Layer {layer_index} Loss: {layer_loss}")

        if method == "Classifier":
            # NOTE: self.learning_rate_cooldown(chapter-1) has to be adjusted in the class function (to get right epoch)
            model.train_BP_Layers(neu_data, torch.tensor(labels, device=DEVICE), chapter=epoch, use_batches=True)

        if epoch % SAVE_STEPS == 0:
            path = f"{save_path}_{epoch}.pth"
            torch.save(model, path)
            print(f"Model saved at {path}")


if __name__ == "__main__":
    model_name = f"{ENV_NAME}_FFNet"
    method = "Classifier"

    # Set up logging
    logging.basicConfig(
        filename=f'{model_name}_{method}.txt',  # Log file name
        level=logging.INFO,  # Log level
        format='Epoch %(epoch)s - Avg Reward: %(avg_total_reward)s',  # Log format
    )
    # create gym environment
    env = gym.make(ENV_NAME)
    input_dim = env.observation_space.shape[0] # 8 for LunarLander
    output_dim = env.action_space.n  # 4 for LunarLander

    # Initialize the FFNet model
    ff_model = FFNet(dims=[input_dim + output_dim] + HIDDEN_DIMS, output_dim=output_dim, use_classifier=method=="Classifier")
    train_ff_model(model=ff_model, env=env, epochs=EPOCHS, n=N_TRAJ_SAMPLES, max_steps=MAX_STEPS, gamma=GAMMA,
                   thresholds=STATE_DIM_THRESHOLDS, method="Goodness", save_path=os.path.join('..', 'models', model_name + method + 'pairwise'))


