# -*- coding: utf-8 -*-
import torch
import pandas as pd
from helper import argmax, argmin, MNIST_loaders, CIFAR10_loaders, process_data, get_data, get_model_name
from networks import FFNet
from agent import LunarLander
from display import LunarLanderDisplay

def trainFF_MNIST(learning_rate=0.01, threshold_coeff=0.01, adaptive_x_neg=False, epochs=100,
                  use_classifier=False, use_regressor=False, lr_cooldown=True,
                  split=None, seed=42, randomize_each_chapter=False, batch_size=64):
    if split is None:
        split = epochs
    torch.manual_seed(seed)
    train_loader_all, test_loader_all = MNIST_loaders(
        train_batch_size=60000, test_batch_size=10000)
    x_train_pos, x_train_neg, x_train_neu, y_train, x_test, x_test_neu, y_test, total_size = get_data(
        train_loader_all, test_loader_all)
    torch.manual_seed(seed)
    model_name = get_model_name(split, learning_rate, threshold_coeff,
                                adaptive_x_neg, use_classifier, use_regressor, randomize_each_chapter)
    net = FFNet(dims=[784, 2000, 2000, 2000, 2000], model_name=model_name,
                use_classifier=use_classifier, batch_size=batch_size, output_dim=10)
    net.train_net(x_train_pos, x_train_neg, x_train_neu, y_train, x_test, x_test_neu, y_test, total_size,
                  epochs=epochs, adaptive_x_neg=adaptive_x_neg, split=split,
                  randomize_each_chapter=randomize_each_chapter, lr_cooldown=lr_cooldown)

def trainFF_CIFAR(learning_rate=0.01, threshold_coeff=0.01, adaptive_x_neg=False, epochs=100,
                  use_classifier=False, use_regressor=False, lr_cooldown=True,
                  split=None, seed=42, randomize_each_chapter=False, batch_size=64):
    if split is None:
        split = epochs
    torch.manual_seed(seed)
    train_loader_all, test_loader_all = CIFAR10_loaders(
        train_batch_size=50000, test_batch_size=10000)
    x_train_pos, x_train_neg, x_train_neu, y_train, x_test, x_test_neu, y_test, total_size = get_data(
        train_loader_all, test_loader_all)
    model_name = get_model_name(split, learning_rate, threshold_coeff,
                                adaptive_x_neg, use_classifier, use_regressor, randomize_each_chapter)
    torch.manual_seed(seed)
    net = FFNet(dims=[3072, 2000, 2000, 2000, 2000], model_name=model_name,
                use_classifier=use_classifier, batch_size=batch_size)
    net.train_net(x_train_pos, x_train_neg, x_train_neu, y_train,
                  x_test, x_test_neu, y_test, total_size,
                  epochs=epochs, adaptive_x_neg=adaptive_x_neg, split=split,
                  randomize_each_chapter=randomize_each_chapter, lr_cooldown=lr_cooldown)

def trainFF_HOUSE(learning_rate=0.01, threshold_coeff=0.01, adaptive_x_neg=False, epochs=100,
                  use_classifier=False, use_regressor=False, lr_cooldown=True, bins=10,
                  split=None, seed=42, randomize_each_chapter=False, batch_size=64):
    if split is None:
        split = epochs
    df = pd.read_csv('housing.csv')
    x_columns = ['Avg. Area Income', 'Avg. Area House Age',
                 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']
    y_column = 'Price'
    (x_train_pos, x_train_neg, x_train_neu, y_train, y_train_num,
     x_test, x_test_neu, y_test, y_test_num, total_size) = process_data(df, x_columns, y_column, quantiles=bins)
    model_name = get_model_name(split, learning_rate, threshold_coeff,
                                adaptive_x_neg, use_classifier, use_regressor, randomize_each_chapter)
    torch.manual_seed(seed)
    net = FFNet(dims=[len(x_train_pos[0]), 2000, 2000, 2000, 2000], model_name=model_name, output_dim=bins,
                use_classifier=use_classifier, use_regressor=use_regressor, batch_size=batch_size)
    net.train_net(x_train_pos, x_train_neg, x_train_neu, y_train,
                  x_test, x_test_neu, y_test, total_size, y_train_num=y_train_num, y_test_num=y_test_num,
                  epochs=epochs, adaptive_x_neg=adaptive_x_neg, split=split,
                  randomize_each_chapter=randomize_each_chapter, lr_cooldown=lr_cooldown)

def sim():
    env = LunarLander()
    display = LunarLanderDisplay(env)
    best_action = 0
    for _ in range(100):
        display.render(best_action)
        rewards = []
        for a in env.action_space:
            env_copy = LunarLander(env)
            observation, reward, done = env_copy.step(a)
            print(a, observation, reward, done)
            rewards.append(reward)
        best_action, best_reward = argmax(rewards)
        print("Best Action:", best_action, "Best Reward:", best_reward)
        worst_action, worst_reward = argmin(rewards)
        print("Worst Action:", worst_action, "Worst Reward:", worst_reward)
        observation, reward, done = env.step(best_action)
        print(observation, reward, done)
        if done:
            print("DONEEEEEEEEEEEEEEEE")
            observation = env.reset()
    display.render(best_action)
    display.close()