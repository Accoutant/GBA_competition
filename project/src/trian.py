from torch import nn, optim
from net import linear_net, LSTMWithLinear, SelfAttention, Bertblock, Bert, BertwithLstm, LstmWithTransformer
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import TensorDataset, DataLoader
from process_data import get_k_fold_data


device = d2l.try_gpu()


# net = linear_net
# net = LSTMWithLinear(27, 20, 32, 3, 3, dropout=0.1)
# net = SelfAttention(27, num_heads=3, dropout=0.2, key_size=27, value_size=27, output_features=3, hidden_size=32)
net = Bert(in_features=11, hidden_size=16, num_heads=4, dropout=0, out_features=3, num_layers=2, num_steps=60)
# net = BertwithLstm(in_features=27, hidden_size=20, num_heads=4, dropout=0, out_features=3, bert_layers=1, lstm_layers=1, num_steps=60)
# net = LstmWithTransformer(in_features=27, hidden_size=20, lstm_layers=2, tf_layers=1, dropout=0, num_heads=4, out_features=3)

# 加载数据
with open("train_data.pkl", "rb") as f:
    train_iter, test_iter = pickle.load(f)
loss_fn = nn.MSELoss()
lr = 0.01
new_lr = 0.01
max_epochs = 10
optimizer = optim.Adam


def trainer(net, train_iter, test_iter, loss_fn, optimizer, max_epochs, device, lr, new_lr):
    net = net.to(device)
    animator = d2l.Animator(xlabel="epoch", ylabel="loss", legend=['train_loss', "test_loss"])
    for epoch in range(max_epochs):
        optim = optimizer(net.parameters(), lr=lr)
        metric = d2l.Accumulator(4)
        iter = 1
        for X, Y in train_iter:
            X = X.to(device)
            Y = Y.to(device)
            output = net(X).squeeze(-1)
            loss = loss_fn(output, Y).mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optim.step()
            metric.add(1, loss, 0, 0)
            print('| epoch %d | iter %d | loss %.4f |' % (epoch+1, iter, metric[1]/metric[0]))
            iter += 1

        for X_test, Y_test in test_iter:
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)
            with torch.no_grad():
                output_test = net(X_test).squeeze(-1)
                loss_test = loss_fn(output_test, Y_test)
                metric.add(0, 0, 1, loss_test)

        print('acc : %.4f, lr: %.4f' % (metric[3]/metric[2], lr))
        animator.add(epoch+1, [metric[1]/metric[0], metric[3]/metric[2]])

        if metric[3]/metric[2] < 3.5:
            lr = new_lr
    torch.save(net.state_dict(), "params.pkl")
    plt.savefig(fname="../models/loss.jpg")


class Trainer(nn.Module):
    def __init__(self, net, loss, optimizer, lr, device):
        super().__init__()
        self.net = net
        self.loss = loss
        self.lr = lr
        self.optimizer = optimizer(self.net.parameters(), lr=self.lr)
        self.device = device
        self.net = self.net.to(self.device)

    def fit(self, train_iter, X_test, Y_test, max_epochs):
        metric = d2l.Accumulator(4)
        for epoch in range(max_epochs):
            iter = 1
            for X, Y in train_iter:
                X = X.to(self.device)
                Y = Y.to(self.device)
                output = self.net(X).squeeze(-1)
                loss = self.loss(output, Y).mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                self.optimizer.step()
                metric.add(1, loss, 0, 0)
                print('| epoch %d | iter %d | loss %.4f |' % (epoch + 1, iter, metric[1] / metric[0]))
                iter += 1

            X_test = X_test.to(self.device)
            Y_test = Y_test.to(self.device)
            with torch.no_grad():
                output_test = self.net(X_test).squeeze(-1)
                loss_test = self.loss(output_test, Y_test)
                metric.add(0, 0, 1, loss_test)

            print('acc : %.4f, lr: %.4f' % (metric[3] / metric[2], self.lr))
            # animator.add(epoch + 1, [metric[1] / metric[0], metric[3] / metric[2]])
        return metric[1] / metric[0], metric[3] / metric[2]


def k_fold_train(k, x, y, trainer, max_epochs, batch_size, device):
    """
    k折交叉验证
    :param k: 折数
    :param x: 打乱后的特征数据
    :param y: 打乱后的标签数据
    :param trainer: 训练器，trainer.fit(self, train_iter, test_features, test_labels, max_epochs, device)
    :param max_epochs: epochs
    :param batch_size: batch_size
    :param device: device
    :return: none
    """
    trainer = trainer
    animator = d2l.Animator(xlabel="k", ylabel="loss", legend=['train_loss', "test_loss"])
    for epoch in range(max_epochs):
        for i in range(k):
            print('-'*25, 'k_fold: %d' % (i+1), '-'*25)
            # 得到训练数据
            (x_train, y_train), (x_test, y_test) = get_k_fold_data(k, i+1, x, y)
            # 将numpy转为tensor
            x_train, y_train = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
            x_test, y_test = torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
            train_dataset = TensorDataset(x_train, y_train)
            train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            train_loss, test_loss = trainer.fit(train_iter, x_test, y_test, 1)
            animator.add(epoch+1+(i+1)/(k+1), [train_loss, test_loss])
    torch.save(trainer.net.state_dict(), "params.pkl")
    plt.savefig(fname="../models/loss.jpg")


print("\n", next(iter(train_iter))[0].shape)
trainer(net, train_iter, test_iter, loss_fn, optimizer, max_epochs=max_epochs, device=device, lr=lr, new_lr=new_lr)
"""
with open("train_data.pkl", "rb") as f:
    X_data, Y_data = pickle.load(f)

loss = nn.MSELoss()
optimizer = optim.Adam
lr = 0.01
max_epochs = 2
batch_size = 128

k_trainer = Trainer(net, loss, optimizer, lr, device)
k_fold_train(k=8, x=X_data, y=Y_data, trainer=k_trainer, max_epochs=max_epochs, batch_size=batch_size, device=device)
"""