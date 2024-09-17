import os
import time 
import torch
import random
import threading
import gymnasium as gym
from collections import deque
from networks import RLNet, FFNet, DDPGNet, ActorCriticNet
from helper import overlay_y_on_x, overlay_y_on_x_single

def train_ff_model_imit(env, mentor_model, ff_model: FFNet, epochs=10, buffer_capacity=20000, buffer_replace_percent=0.4, use_classifier=False):
    pos_buffer = deque(maxlen=buffer_capacity)
    neg_buffer = deque(maxlen=buffer_capacity)
    neu_buffer = deque(maxlen=buffer_capacity)
    labels     = deque(maxlen=buffer_capacity)
    batch_size = 32
    ff_model.epochs = epochs
    ff_model.split  = epochs
    epoch = 0
    while epoch <= epochs:
        state, info = env.reset()
        done = False
        losses = []
        while not done:
            state_ = torch.FloatTensor(state).cuda()
            ff_input = overlay_y_on_x(state_, neu=True, output_dim=output_dim, add=True, single_item=True)
            action_ff = ff_model.predict(ff_input, method = "Goodness").item()
            action_bp = mentor_model.single_predict(state_)
            if int(action_bp) != int(action_ff):
                pos_state = overlay_y_on_x(ff_input, [action_bp], False, output_dim)[0]
                neg_state = overlay_y_on_x(ff_input, [action_ff], False, output_dim)[0]
                pos_buffer.append(pos_state)
                neg_buffer.append(neg_state)
                neu_buffer.append(ff_input[0])
                labels.append(action_bp)
            state, reward, terminated, truncated, info = env.step(action_ff)
            done = terminated or truncated
        if min(len(pos_buffer),len(neg_buffer)) >= buffer_capacity * buffer_replace_percent:
            ff_model.learning_rate_cooldown(ff_model.current_epoch(epoch, 0))
            indices = random.sample(range(len(pos_buffer)), batch_size)
            input_neu = torch.stack([neu_buffer[i] for i in indices], dim=0)
            input_pos = torch.stack([pos_buffer[i] for i in indices], dim=0)
            input_neg = torch.stack([neg_buffer[i] for i in indices], dim=0)
            input_labels = torch.tensor([labels[i] for i in indices], dtype=torch.long).cuda()
            for layer_index, layer in enumerate(ff_model.layers):
                ff_model.learning_rate_cooldown(ff_model.current_epoch(epoch, 0))
                losses.append(layer.train_layer(input_pos, input_neg))
                if layer_index != len(ff_model.layers)-1:
                    input_pos, input_neg = layer.output(input_pos), layer.output(input_neg)
            if use_classifier:
                ff_model.train_BP_Layers(x_train_neu=input_neu, y_train=input_labels, chapter=epoch)
            epoch += 1
            if epoch > 0 and epoch % 50 == 0:
                print(f'Epoch {epoch}, Avg. Loss: {sum(losses)/len(losses)}')
        if epoch>0 and epoch%100 == 0:
            good_thread = threading.Thread(target=test_model, args=(ff_model, "ff", env_name, 5, "Goodness"))
            good_thread.start()
            if use_classifier:
                class_thread = threading.Thread(target=test_model, args=(ff_model, "ff", env_name, 5, "Classifier"))
                class_thread.start()
        if epoch > 0 and epoch % 1000 == 0:
            torch.save(ff_model, model_path_ff)

def train_ff_model_ss(env, ff_model: FFNet, epochs=10, buffer_capacity=1000, buffer_replace_percent=0.4):
    pos_buffer = deque(maxlen=buffer_capacity)
    neg_buffer = deque(maxlen=buffer_capacity)
    batch_size = 32
    ff_model.epochs = epochs
    ff_model.split = epochs
    epoch = 0
    reward_history = deque(maxlen=500)  # Moving window for reward history
    moving_average_reward = 0
    while epoch <= epochs:
        state, info = env.reset()
        done = False
        losses = []
        while not done:
            state_ = torch.FloatTensor([state]).cuda()
            ff_input = overlay_y_on_x(state_, neu=True, output_dim=output_dim, add=True)
            action_ff = ff_model.predict(ff_input, method="Goodness").item()
            next_state, reward, terminated, truncated, info = env.step(action_ff)
            reward_history.append(reward)
            if len(reward_history) > 100:
                moving_average_reward = sum(reward_history) / len(reward_history)
                if reward > moving_average_reward:
                    pos_state = overlay_y_on_x(state_, [action_ff], False, output_dim, True)[0]
                    pos_buffer.append(pos_state)
                else:
                    neg_state = overlay_y_on_x(state_, [action_ff], False, output_dim, True)[0]
                    neg_buffer.append(neg_state)
            if min(len(pos_buffer), len(neg_buffer)) >= buffer_capacity * buffer_replace_percent:
                indices = random.sample(range(min(len(pos_buffer), len(neg_buffer))), batch_size)
                input_pos = torch.stack([pos_buffer[i] for i in indices], dim=0)
                input_neg = torch.stack([neg_buffer[i] for i in indices], dim=0)
                for layer_index, layer in enumerate(ff_model.layers):
                    ff_model.learning_rate_cooldown(ff_model.current_epoch(epoch, 0))
                    losses.append(layer.train_layer(input_pos, input_neg))
                    if layer_index != len(ff_model.layers) - 1:
                        input_pos, input_neg = layer.output(input_pos), layer.output(input_neg)
            state = next_state
            done = terminated or truncated
        if min(len(pos_buffer), len(neg_buffer)) >= buffer_capacity * buffer_replace_percent:
            epoch += 1
        if epoch > 0 and epoch % 100 == 0:
            print(f'Epoch {epoch}, Avg. Loss: {sum(losses)/len(losses)}, Moving Avg Reward: {moving_average_reward}')
            
def test_model(model, model_type, env_name, epochs=5, mode = "Goodness"):
    print(f"Model: {model_type}, Env: {env_name}", "Mode:", mode)
    test_env  = gym.make(env_name, render_mode = "human")
    total_reward = 0
    for i in range(epochs):
        state, info = test_env.reset()
        done = False
        counter = 0
        while not done:
            counter += 1
            test_env.render()
            state = torch.FloatTensor(state).cuda()
            if model_type == "ff":
                ff_input = overlay_y_on_x(state, neu=True, output_dim=output_dim, add=True, single_item=True)
                action = model.predict(ff_input, method = mode).item()
            else:
                action = model(state)
            state, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward
            done = terminated or truncated
        #print(f"Epoch {i}, Steps: {counter}", end="-")
    print(f"Total Reward: {total_reward}, Epochs {epochs}, Avg. Reward: {total_reward/epochs}")
    test_env.close()

env_name = 'MountainCar-v0'
env_name = 'FlappyBird-v0'
env_name = "CartPole-v1"
env_name = "LunarLander-v2"
hidden_dims = [128, 256, 128]
hidden_dims = [128] 

train_env = gym.make(env_name)
input_dim = train_env.observation_space.shape[0]
output_dim = train_env.action_space.n
print("Input Dim:", input_dim, "Output Dim:", output_dim)
models_path = "models/"
if not os.path.exists(models_path):
    os.makedirs(models_path)
model_path_bp = f"./{models_path}{env_name}_BPNet.pth"
model_path_ff = f"./{models_path}{env_name}_FFNet.pth"
model_path_ddpg = f"./{models_path}{env_name}_DDPGNet.pth"
model_path_actor_critic = f"./{models_path}{env_name}_ActorCriticNet.pth"
mentor = "ddpg"
mentor = "actor_critic"
mentor_model = None

if mentor == "bp":
    if os.path.exists(model_path_bp):
        bp_model = torch.load(model_path_bp)
    else:
        bp_model = RLNet(dims=[input_dim] + hidden_dims + [output_dim], learning_rate=0.0001)
        bp_model.train_rl_model_policy_gradient(train_env, num_episodes=1000)
        torch.save(bp_model, model_path_bp)
    test_model(bp_model, "bp", env_name)
    mentor_model = bp_model
elif mentor == "ddpg":
    if os.path.exists(model_path_ddpg):
        ddpg_model = torch.load(model_path_ddpg)
    else:
        ddpg_model = DDPGNet(state_dim=input_dim, action_dim=output_dim, hidden_dims=hidden_dims, 
                             actor_lr=0.00001, critic_lr=0.0001)
        ddpg_model.train_ddpg_model(train_env, num_episodes=3000, batch_size=128)
        torch.save(ddpg_model, model_path_ddpg)
    test_model(ddpg_model, "ddpg", env_name)
    mentor_model = ddpg_model
elif mentor == "actor_critic":
    if os.path.exists(model_path_actor_critic):
        actor_critic_model = torch.load(model_path_actor_critic, weights_only=False)
    else:
        actor_critic_model = ActorCriticNet(state_dim=input_dim, action_dim=output_dim, hidden_dims=hidden_dims,
                                            learning_rate=0.01)
        actor_critic_model.train_actor_critic_model(train_env, num_episodes=3000)
        torch.save(actor_critic_model, model_path_actor_critic)
    #test_model(actor_critic_model, "actor_critic", env_name)
    mentor_model = actor_critic_model
    
def train_ff_model(env_name, model_path_ff, input_dim, output_dim, hidden_dims, use_classifier, mentor, mentor_model):
    train_env = gym.make(env_name)
    if os.path.exists(model_path_ff):
        ff_model = torch.load(model_path_ff)
    else:
        ff_model = FFNet(dims=[input_dim + output_dim] + hidden_dims, output_dim=output_dim, use_classifier=use_classifier)
        ff_model.train()
        if mentor == "self":
            train_ff_model_ss(train_env, ff_model, epochs=3000)
        else:
            train_ff_model_imit(train_env, mentor_model, ff_model, epochs=3000, use_classifier=use_classifier)
        torch.save(ff_model, model_path_ff)
    return ff_model

train_env = gym.make(env_name)
use_classifier = True
hidden_dims = [2000, 2000, 2000, 2000]
ff_model = train_ff_model(env_name, model_path_ff, input_dim, output_dim, hidden_dims, 
                          use_classifier, mentor, mentor_model)
