import torch
import torch.nn as nn
from torch.optim import Adam

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
print(DEVICE)


class FFLayer(nn.Linear):
    def __init__(self, in_features, out_features, learning_rate=0.03, threshold=2.0):
        super().__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.opt = Adam(self.parameters(), lr=learning_rate)
        self.activation = torch.nn.ReLU()
        self.threshold = threshold

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.activation(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train_layer(self, x_pos, x_neg):
        self.opt.zero_grad()
        g_pos = self.forward(x_pos).pow(2).mean(1)
        g_neg = self.forward(x_neg).pow(2).mean(1)
        loss = torch.log(
            1 + torch.exp(torch.cat([-g_pos + self.threshold, g_neg - self.threshold]))).mean()
        loss.backward()
        self.opt.step()
        return loss.item()

    def output(self, x):
        return self.forward(x).detach()


class FFClassifier(nn.Linear):
    def __init__(self, input_dim, output_dim, lr, criterion):
        super().__init__(input_dim, output_dim, bias=False, device=DEVICE)
        self.lr = lr
        self.criterion = criterion()
        self.opt = Adam(self.parameters(), lr=self.lr)

    def train_layer(self, x, y):
        self.opt.zero_grad()
        output = self.forward(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.opt.step()


class FFRegressor(nn.Linear):
    def __init__(self, input_dim, output_dim, lr, criterion):
        super().__init__(input_dim, output_dim, bias=False, device=DEVICE)
        self.lr = lr
        self.criterion = criterion()
        self.opt = Adam(self.parameters(), lr=self.lr)

    def train_layer(self, x, y):
        self.opt.zero_grad()
        output = self.forward(x).squeeze()
        loss = self.criterion(output, y)
        loss.backward()
        self.opt.step()


class BPLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))
