import random
from collections import OrderedDict
from itertools import combinations
from functools import partial
import numpy as np
import torch
import gymnasium as gym
from networks import FFNet
from helper import overlay_y_on_x, move
import os
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
BUFFER_LIMIT = 30000
SAMPLES_PER_EPOCH = 500
# For LunarLander the range of 8-dimensional state space is: [(-1.5, 1.5), (-1.5, 1.5), (-5, 5), (-5, 5), (-3.14, 3.14), (-5, 5), {0, 1}, {0, 1}]
# chosen, so that each dimension has 100 discrete steps
STATE_DIM_THRESHOLDS = [0.03, 0.03, 0.1, 0.1, 0.0628, 0.1, 1, 1]
GAMMA = 0.99

LOG_STEPS = 50
SAVE_STEPS = 500

###################
# Data Generation #
###################


class LimitedDict(OrderedDict):
    def __init__(self, size_limit):
        super().__init__()
        self.size_limit = size_limit

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        while len(self) > self.size_limit:
            self.popitem(last=False)


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
def bin_states(bins, trajectories, thresholds, nb_actions=4):
    """
    Bin states across all trajectories based on a threshold per state dimension.

    Parameters:
    - bins: Instance of LimitedDict
    - trajectories: List of trajectories, where each trajectory is a list of (state, action, reward, long-term reward) tuples
    - thresholds: List of threshold values for each state dimension
    - nb_actions: Number of actions for current environment

    Returns:
    - bins: A dictionary where each key is a bin (tuple representing bin index) and the value is a list of states in that bin
    """

    for trajectory in trajectories:
        for state, action, reward, lt_reward in trajectory:
            # perform rounded division by threshold to get bin index
            bin_index = tuple((state // thresholds).astype(int))

            # for every explored state provide space for every possible action
            if bin_index not in bins:
                # ToDo: 'state' entry just needs to be np.zeros(len(state)), because bin entry is for specific state.
                #  This means that we don't have to differentiate in other functions (but just inefficient not wrong)
                bins[bin_index] = {'state': np.zeros((nb_actions, len(state))),
                                   'lt_reward': np.zeros(nb_actions), 'explored': np.zeros(nb_actions)}
            # only store if new lt_reward is better than previous entry
            if bins[bin_index]['explored'][action] == 0 or bins[bin_index]['lt_reward'][action] < lt_reward:
                bins[bin_index]['state'][action] = state
                bins[bin_index]['lt_reward'][action] = lt_reward
                bins[bin_index]['explored'][action] = 1

    return bins

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


def train_ff_model(model, env, epochs,  n, max_steps, gamma, thresholds, buffer_limit=1000, samples_per_epoch=500,
                   method="Goodness", save_path="ff_model"):
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
    - buffer_limit: Maximum number of states to store in the buffer
    - samples_per_epoch: Number of samples to collect before each epoch
    - method: Method for FF Training (Goodness | Classifier)
    - save_path: Path to save the model
    """

    model.train()  # Set the model to training mode
    model.epochs = epochs
    buffer = LimitedDict(buffer_limit)
    print(f"Epoch 1/{epochs}")
    for epoch in range(1, epochs + 1):
        if epoch % LOG_STEPS == 0:
            print(f"Epoch {epoch}/{epochs}")
        # wrap current model into a policy (input a state and output action probabilities
        current_policy = partial(get_action_probs, ffnet=model, method=method)
        while True:
            # sample trajectories
            trajectories, avg_total_reward = sample_trajectories(env, current_policy, n, max_steps, gamma)

            # bin states
            buffer = bin_states(buffer, trajectories, thresholds, nb_actions=env.action_space.n)
            relevant_entries = {key: value for key, value in buffer.items() if np.sum(value['explored']) > 1}
            if len(relevant_entries) > samples_per_epoch:
                break
        if epoch % LOG_STEPS == 0:
            print(f"Average Total Reward per Episode: {avg_total_reward}")
            logging.info('', extra={'epoch': epoch, 'avg_total_reward': avg_total_reward})
        candidates = random.sample(relevant_entries.items(), samples_per_epoch)
        pos_states = []
        pos_actions = []
        neg_states = []
        neg_actions = []

        # create positive and negative data by iterating over bin buffer and taking min and max actions
        for key, candidate in candidates:
            true_indices = np.nonzero(candidate['explored'])[0]
            high_index = true_indices[np.argmax(candidate['lt_reward'][true_indices])]
            low_index = true_indices[np.argmin(candidate['lt_reward'][true_indices])]

            high_state = candidate['state'][high_index]
            low_state = candidate['state'][low_index]

            pos_states.append(high_state)
            pos_actions.append(high_index)
            neg_states.append(low_state)
            neg_actions.append(low_index)
        if epoch % LOG_STEPS == 0:
            print(f"Number of positive/negative data points: {len(pos_states)}")
        if len(pos_states) == 0:
            print("No positive/negative data points found. Skipping epoch.")
            continue
        # construct training input with overlay
        pos_data = overlay_y_on_x(move(np.array(pos_states)), pos_actions, neu=False, output_dim=model.output_dim, add=True)
        neg_data = overlay_y_on_x(move(np.array(neg_states)), neg_actions, neu=False, output_dim=model.output_dim, add=True)

        neu_data = overlay_y_on_x(move(np.array(pos_states)), neu=True, output_dim=model.output_dim, add=True)
        labels = pos_actions

        model.total_size = len(pos_data)
        for layer_index, layer in enumerate(model.layers):
            layer_loss = 0
            model.learning_rate_cooldown(epoch)
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
                   thresholds=STATE_DIM_THRESHOLDS, buffer_limit=BUFFER_LIMIT, samples_per_epoch=SAMPLES_PER_EPOCH,
                   method=method, save_path=os.path.join('..', 'models', model_name + method + 'buffer'))


