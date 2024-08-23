import os
import torch
import gymnasium as gym
from networks import BPNet, FFNet
from helper import overlay_y_on_x


env_name = "CartPole-v1"
hidden_dims = [1000, 1000, 1000]  # Set hidden layers as 3x1000

# Initialize Gymnasium environment
env = gym.make(env_name)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# Define model saving/loading path
model_path_bp = f"./models/{env_name}_BPNet.pth"
model_path_ff = f"./models/{env_name}_FFNet.pth"

# Load or create a BPNet model
if os.path.exists(model_path_bp):
    bp_model = torch.load(model_path_bp)
else:
    bp_model = BPNet(dims=[input_dim] + hidden_dims + [output_dim])
    bp_model.train_rl_model(env)  # Placeholder for actual training logic
    torch.save(bp_model, model_path_bp)

# Initialize the FFNet model
ff_model = FFNet(dims=[input_dim + output_dim] + hidden_dims)
ff_model.train()  # Set the model to training mode

def train_ff_model(env, bp_model: BPNet, ff_model: FFNet, epochs=10, buffer_capacity=1000, buffer_replace_percent=0.4):
    from collections import deque
    import random

    pos_buffer = deque(maxlen=buffer_capacity)
    neg_buffer = deque(maxlen=buffer_capacity)

    for epoch in range(epochs):
        state, info = env.reset()
        done = False
        while not done:
            state_ = torch.FloatTensor([state]).cuda()
            ff_input  = overlay_y_on_x(state_, neu=True, output_dim=output_dim, add=True)
            action_ff = ff_model.predict(ff_input, method = "Goodness").item()
            action_bp = bp_model.predict(state_).item()
            print(action_bp, action_ff)
            if action_bp != action_ff:
                pos_state = overlay_y_on_x(state, action_bp, False, output_dim, True)
                neg_state = overlay_y_on_x(state, action_ff, False, output_dim, True)
                pos_buffer.append((pos_state, action_bp))
                neg_buffer.append((neg_state, action_ff))
            # Check if buffers are sufficiently filled to start training
            if len(pos_buffer) >= buffer_capacity * buffer_replace_percent and len(neg_buffer) >= buffer_capacity * buffer_replace_percent:
                # Sample from buffers and train
                input_pos = random.sample(pos_buffer, int(len(pos_buffer) * buffer_replace_percent))
                input_neg = random.sample(neg_buffer, int(len(neg_buffer) * buffer_replace_percent))
                for layer_index, layer in enumerate(ff_model.layers):
                    layer_loss = 0
                    ff_model.learning_rate_cooldown(ff_model.current_epoch(epoch, 0))
                    layer_loss += layer.train_layer(input_pos, input_neg)
                    if layer_index != len(ff_model.layers)-1:
                        input_pos, input_neg = layer.output(input_pos), layer.output(input_neg)
            state, reward, terminated, truncated, info = env.step(action_bp)
            done = terminated or truncated

# Example of training the FF model
train_ff_model(env, bp_model, ff_model, epochs=50)

# Save the FF model
torch.save(ff_model, model_path_ff)