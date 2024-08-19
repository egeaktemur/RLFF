import torch
import numpy as np
import pandas as pd
from torchvision.datasets import MNIST, CIFAR10
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
print(DEVICE)


def argmax(arr: list) -> int:
    max_value = arr[0]
    max_index = 0
    for i, value in enumerate(arr):
        if value > max_value:
            max_value = value
            max_index = i
    return max_index, max_value


def argmin(arr: list) -> int:
    min_value = arr[0]
    min_index = 0
    for i, value in enumerate(arr[1:]):
        if value < min_value:
            min_value = value
            min_index = i
    return min_index, min_value


def MNIST_loaders(train_batch_size=64, test_batch_size=1000):
    transform = Compose([ToTensor(), Normalize(
        (0.1307,), (0.3081,)), Lambda(lambda x: torch.flatten(x))])
    train_loader = DataLoader(MNIST('./data/', train=True, download=True, transform=transform),
                              batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(MNIST('./data/', train=False, download=True, transform=transform),
                             batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def CIFAR10_loaders(train_batch_size=64, test_batch_size=1000):
    transform = Compose([ToTensor(), Lambda(lambda x: torch.flatten(x))])
    train_loader = DataLoader(CIFAR10('./data/', train=True, download=True, transform=transform),
                              batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(CIFAR10('./data/', train=False, download=True, transform=transform),
                             batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def get_data(train_loader, test_loader):
    x, y_train_numeric = next(iter(train_loader))
    x, y_train_numeric = x.to(DEVICE), y_train_numeric.to(DEVICE)
    x_train_pos = overlay_y_on_x(x, y_train_numeric)
    x_train_neg = randomlyGenerateX_neg(x, y_train_numeric)
    x_train_neu = overlay_y_on_x(x, neu=True)
    x_test, y_test_numeric = next(iter(test_loader))
    x_test, y_test_numeric = x_test.to(DEVICE), y_test_numeric.to(DEVICE)
    x_test_neu = overlay_y_on_x(x_test, neu=True)
    total_size = len(x)
    return x_train_pos, x_train_neg, x_train_neu, y_train_numeric, x_test, x_test_neu, y_test_numeric, total_size


def randomlyGenerateX_neg(x, y):
    num_classes = y.max() + 1
    random_labels = torch.randint(0, num_classes - 1, y.shape).to(DEVICE)
    adjust_mask = random_labels >= y
    random_labels += adjust_mask.long()
    return overlay_y_on_x(x, random_labels, False, num_classes).to(DEVICE)


def overlay_y_on_x(x, y=None, neu=False, output_dim=10, add = False):
    x_ = x.clone()
    if add:
      x_ = np.concatenate((x, np.zeros(output_dim)), axis=0)
    if neu:
        x_[:, -output_dim:] = 1/output_dim
    else:
        x_[:, -output_dim:] = 0.0
        y_indices = y + (x.shape[1] - output_dim)
        x_[range(x.shape[0]), y_indices] = 1.0
    return x_


def move(x):
    return torch.tensor(x, dtype=torch.float, device=DEVICE)


def process_data(df, x_columns, y_column, quantiles=10, device=DEVICE):
    scaler = StandardScaler()
    df[x_columns + [y_column]
       ] = scaler.fit_transform(df[x_columns + [y_column]])
    df = df[x_columns + [y_column]]
    encoder = OneHotEncoder(sparse_output=False)
    df['Bins'] = pd.qcut(df[y_column], q=quantiles,
                         labels=False, duplicates='drop')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    y_train, y_test = train_df['Bins'], test_df['Bins']
    y_train_enc, y_test_enc = encoder.fit_transform(
        y_train.values.reshape(-1, 1)), encoder.transform(y_test.values.reshape(-1, 1))
    y_train, y_train_num = move(y_train.values).long(), move(
        train_df[y_column].values)
    y_test, y_test_num = move(y_test.values).long(), move(
        test_df[y_column].values)

    x_train = move(np.hstack([train_df[x_columns].values, y_train_enc]))
    x_test = move(np.hstack([test_df[x_columns].values, y_test_enc]))
    # x_train_neg = generate_negative(x_train, move(y_train_enc), hardness=0.5)
    x_train_neg = randomlyGenerateX_neg(x_train, y_train)
    x_train_pos = overlay_y_on_x(x_train, y_train, num=quantiles)
    x_train_neu = overlay_y_on_x(x_train, neu=True, num=quantiles)
    x_test_neu = overlay_y_on_x(x_test,  neu=True, num=quantiles)
    total_size = len(train_df)
    return (x_train_pos, x_train_neg, x_train_neu, y_train, y_train_num, x_test, x_test_neu, y_test, y_test_num, total_size)


def get_model_name(split, learning_rate, threshold_coeff, adaptive_x_neg, use_classifier, use_regressor, randomize_each_chapter):
    model_name = "ff"
    model_name += '_th'+str(threshold_coeff)
    model_name += "_lr"+str(learning_rate)
    model_name += "_split"+str(split)
    if adaptive_x_neg:
        model_name += "_AdaptiveXNEG"
    if randomize_each_chapter:
        model_name += "_RandomXNEG"
    if use_classifier:
        model_name += "_Classifier"
    if use_regressor:
        model_name += "_Regressor"
    return model_name
