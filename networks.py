import time
import torch
import torch.nn as nn
import gymnasium as gym
from torch.optim import Adam
from torch.distributions import Categorical
from layers import BPLayer, FFLayer, FFClassifier, FFRegressor
from helper import overlay_y_on_x, randomlyGenerateX_neg

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
print(DEVICE)


class BPNet(nn.Module):
    def __init__(self, dims=[784, 500, 400, 10], learning_rate=0.0003):
        super().__init__()
        self.layers = nn.ModuleList(
            [BPLayer(dims[d], dims[d + 1]).cuda() for d in range(len(dims) - 1)])
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.opt = Adam(self.parameters(), lr=self.learning_rate)
        self.batch_size = 64
        print("Finished creating Backpropagation model with lr =", self.learning_rate)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).argmax(1)

    def evaluate(self, x, y):
        self.eval()
        with torch.no_grad():
            correct = self.predict(x).eq(y).float().sum().item()
        accuracy = correct / len(x) * 100
        return accuracy

    def learning_rate_cooldown(self, epoch):
        learning_rate = 2 * self.learning_rate * \
            (self.epochs - epoch)/self.epochs
        if epoch > self.epochs // 2 and learning_rate > 0:
            for param_group in self.opt.param_groups:
                param_group['lr'] = learning_rate

    def generate_batches(self, multiplier=1):
        batch_size = self.batch_size * multiplier
        for start in range(0, self.total_size, batch_size):
            end = min(start + batch_size, self.total_size)
            yield (start, end)

    def train_net(self, x_train, y_train, x_test, y_test,
                  total_size, epochs=100, split=None, report_every=25, lr_cooldown=True):
        self.split, self.epochs, self.total_size = split, epochs, total_size
        start_time = time.time()
        for epoch in range(epochs):
            self.train()
            self.learning_rate_cooldown(epoch)
            for start_index, end_index in self.generate_batches():
                x_batch = x_train[start_index:end_index]
                y_batch = y_train[start_index:end_index]
                self.opt.zero_grad()
                y_pred = self.forward(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.opt.step()
        train_time = (time.time() - start_time)
        test_ACC = self.evaluate(x_test, y_test)
        print(f'Epoch {epochs}, Test accuracy: {test_ACC}% Time: {train_time}')

    def train_rl_model(self, env, num_episodes=1000, gamma=0.99):
        for episode in range(num_episodes):
            state = env.reset()[0]
            log_probs = []
            rewards = []
            done = False
            while not done:
                state = torch.FloatTensor(state).cuda()
                action_probs = self(state)
                m = Categorical(action_probs)
                action = m.sample()
                state, reward, done, _, _ = env.step(action.item())
                log_probs.append(m.log_prob(action))
                rewards.append(reward)
            discounted_rewards = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                discounted_rewards.insert(0, R)
            discounted_rewards = torch.FloatTensor(discounted_rewards).cuda()
            log_probs = torch.stack(log_probs)
            loss = -log_probs * discounted_rewards
            loss = loss.sum()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if episode % 100 == 0:
                print(f'Episode {episode}, Loss: {loss.item()}')
        env.close()

class FFBaseNet(torch.nn.Module):
    epochs = 0

    def __init__(self, dims=[784, 2000, 2000, 2000, 2000], batch_size=64, output_dim=10,
                 learning_rate=0.01, threshold_coeff=0.01, model_name=""):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.threshold_coeff = threshold_coeff
        self.layers = nn.ModuleList([FFLayer(dims[d], dims[d + 1], learning_rate,
                                    dims[d + 1]*threshold_coeff).to(DEVICE) for d in range(len(dims) - 1)])

    def get_activations(self, x):
        with torch.no_grad():
            activations = []
            h = x
            for index, layer in enumerate(self.layers):
                h = layer.output(h)
                if index > 0:  # Skip the first layer
                    activations.append(h)
            combined_activations = torch.cat(activations, dim=1).detach()
        return combined_activations

    def forward(self, x):
        goodness_per_label = []
        for label_index, label in enumerate(range(self.output_dim)):
            h = overlay_y_on_x(x, label, neu=False, num=self.output_dim)
            h = self.layers[0].output(h)
            goodness = []
            for layer_index, layer in enumerate(self.layers[1:]):
                h = layer.output(h)
                goodness_value = h.pow(2).mean(1, keepdim=True)
                goodness.append(goodness_value)
            concatenated_goodness = torch.cat(goodness, dim=1)
            label_goodness = torch.sum(concatenated_goodness, dim=1)
            goodness_per_label.append(label_goodness)
        result = torch.stack(goodness_per_label, dim=1)
        return result

    def create_neg_batch(self, x, y):
        top2_preds = torch.topk(
            self.get_goodness_per_label(x), 2, dim=1).indices
        x_neg_labels = torch.where(
            top2_preds[:, 0] == y, top2_preds[:, 1], top2_preds[:, 0])
        x_neg_batch = overlay_y_on_x(x, x_neg_labels)
        return x_neg_batch

    def freeze_FF_Layers(self):
        for param in self.layers.parameters():
            param.requires_grad = False

    def unfreeze_FF_Layers(self):
        for param in self.layers.parameters():
            param.requires_grad = True

    def generate_batches(self, multiplier=1):
        batch_size = self.batch_size * multiplier
        for start in range(0, self.total_size, batch_size):
            end = min(start + batch_size, self.total_size)
            yield (start, end)

    def current_epoch(self, chapter, mini_epoch):
        return (chapter*(self.epochs//self.split))+mini_epoch+1


class FFNet(FFBaseNet):
    def __init__(self, dims=[784, 2000, 2000, 2000, 2000], batch_size=64, output_dim=10,
                 learning_rate=0.01, threshold_coeff=0.01, model_name="",
                 use_classifier=False, use_regressor=False,
                 classifier_criterion=nn.CrossEntropyLoss, regressor_criterion=nn.MSELoss):
        super().__init__(dims, batch_size, output_dim,
                         learning_rate, threshold_coeff, model_name)
        self.softmax, self.use_classifier, self.use_regressor = nn.Softmax(
            dim=1), use_classifier, use_regressor
        if use_classifier:
            self.classifier_lr_coeff = 0.01
            self.classifier = FFClassifier(sum(
                dims[2:]), output_dim, learning_rate * self.classifier_lr_coeff, classifier_criterion)
        if use_regressor:
            self.regressor_lr_coeff = 0.01
            self.regressor = FFRegressor(
                sum(dims[2:]), 1, learning_rate * self.regressor_lr_coeff, regressor_criterion)
        print("Finished creating model:", self.model_name)

    def predict(self, x, method="Goodness"):
        with torch.no_grad():
            if method == "Regression":
                return self.regressor(self.get_activations(x))[0]
            if method == "Classifier":
                logits = self.classifier(self.get_activations(x))
            if method == "Goodness":
                logits = self.forward(x)
            probabilities = self.softmax(logits)
            prediction = probabilities.argmax(dim=1)
        return prediction

    def evaluate(self, x, y, method="Goodness"):
        self.eval()
        with torch.no_grad():
            if method == "Regression":
                prediction = self.regressor(self.get_activations(x))
                abs_mean_error = torch.mean(torch.abs(prediction - y))
                return float(abs_mean_error)
            else:
                prediction = self.predict(x, method=method)
                correct = prediction.eq(y).float().sum().item()
        accuracy = int(correct) / len(x) * 100
        return accuracy

    def learning_rate_cooldown(self, epoch, lr_cooldown=True):
        if lr_cooldown:
            learning_rate = 2 * self.learning_rate * \
                (self.epochs - epoch)/self.epochs
            if epoch > self.epochs // 2 and learning_rate > 0:
                for layer in self.layers:
                    layer.opt.param_groups[0]['lr'] = learning_rate
                if self.use_classifier:
                    self.classifier.opt.param_groups[0]['lr'] = learning_rate * \
                        self.classifier_lr_coeff
                if self.use_regressor:
                    self.regressor.opt.param_groups[0]['lr'] = learning_rate * \
                        self.regressor_lr_coeff

    def generate_x_neg(self, x_train_neu, y_train_enc, adaptive_x_neg, randomize_each_chapter):
        if adaptive_x_neg and self.use_classifier:
            top2_preds = torch.topk(self.classifier(
                self.get_activations(x_train_neu)), 2, dim=1).indices
            labels = torch.where(
                top2_preds[:, 0] == y_train_enc, top2_preds[:, 1], top2_preds[:, 0])
            x_train_neg = overlay_y_on_x(
                x_train_neu, labels, False, self.output_dim).to(DEVICE)
        elif adaptive_x_neg and not self.use_classifier:
            for start_index, end_index in self.generate_batches(10):
                x_train_neg[start_index:end_index] = self.create_neg_batch(
                    x_train_neu[start_index:end_index], y_train_enc[start_index:end_index])
        elif randomize_each_chapter:
            x_train_neg = randomlyGenerateX_neg(x_train_neu, y_train_enc)
        return x_train_neg

    def test_model(self, x_test_neu, y_test, y_test_num=None, epochs=0, train_time=0):
        regressor_test_AME, classifier_test_ACC = 0, 0
        output = "Epochs: " + str(epochs) + \
            " Training Time: " + str(train_time)
        goodness_test_ACC = self.evaluate(
            x_test_neu, y_test, method="Goodness")
        output += "\n\tGoodness Test ACC: " + str(goodness_test_ACC)
        if self.use_classifier:
            classifier_test_ACC = self.evaluate(
                x_test_neu, y_test, method="Classifier")
            output += "\n\tClassifier Test ACC: " + str(classifier_test_ACC)
        if self.use_regressor:
            regressor_test_AME = self.evaluate(
                x_test_neu, y_test_num, method="Regression")
            output += "\n\tRegressor Test AME: " + str(regressor_test_AME)
        print(output)

    def train_BP_Layers(self, x_train_neu, y_train, y_train_num, epoch, chapter, mini_epoch):
        input = self.get_activations(x_train_neu)
        for mini_epoch in range(epoch):
            self.learning_rate_cooldown(
                self.current_epoch(chapter, mini_epoch))
            for start_index, end_index in self.generate_batches():
                if self.use_regressor:
                    self.regressor.train_layer(
                        input[start_index:end_index], y_train_num[start_index:end_index])
                if self.use_classifier:
                    self.classifier.train_layer(
                        input[start_index:end_index], y_train[start_index:end_index])

    def train_net(self, x_train_pos, x_train_neg, x_train_neu, y_train,
                  x_test, x_test_neu, y_test, total_size, y_train_num=None, y_test_num=None,
                  epochs=100, split=None, stop_acc=100.0, report_every=25,
                  adaptive_x_neg=False, randomize_each_chapter=False, lr_cooldown=True):
        self.split, self.epochs, self.total_size = split, epochs, total_size
        start_time = time.time()
        for chapter in range(split):
            self.train()
            self.unfreeze_FF_Layers()
            input_pos, input_neg = x_train_pos, x_train_neg
            for layer_index, layer in enumerate(self.layers):
                layer_loss = 0
                for mini_epoch in range(epochs//split):
                    self.learning_rate_cooldown(self.current_epoch(chapter, mini_epoch))
                    for start_index, end_index in self.generate_batches():
                        layer_loss += layer.train_layer(
                            input_pos[start_index:end_index], input_neg[start_index:end_index])
                if layer_index != len(self.layers)-1:
                    input_pos, input_neg = layer.output(
                        input_pos), layer.output(input_neg)
                    # print("Layer "+str(layer_index)+":",layer_loss)
            self.freeze_FF_Layers()
            if self.use_classifier or self.use_regressor:
                self.train_BP_Layers(
                    x_train_neu, y_train, y_train_num, epochs//split, chapter, mini_epoch)
            if adaptive_x_neg or randomize_each_chapter:
                x_train_neg = self.generate_x_neg(
                    x_train_neu, y_train, adaptive_x_neg, randomize_each_chapter)
            if chapter*(epochs//split) % report_every == 0 and chapter*(epochs//split) != self.epochs:
                self.test_model(x_test_neu, y_test, y_test_num,
                                chapter*(epochs//split), time.time() - start_time)
        train_time = (time.time() - start_time)
        self.test_model(x_test_neu, y_test, y_test_num, epochs, train_time)