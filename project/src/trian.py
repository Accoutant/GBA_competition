from torch import nn, optim
from net import linear_net, rmse
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import pickle


device = d2l.try_gpu()

# 加载数据
with open("train_data.pkl", "rb") as f:
    train_iter, test_iter = pickle.load(f)

net = linear_net
loss_fn = nn.MSELoss()
lr = 0.001
max_epochs = 25
optimizer = optim.SGD(net.parameters(), lr=lr)


def trainer(net, train_iter, test_iter, loss_fn, optimizer, max_epochs, device):
    net = net.to(device)
    animator = d2l.Animator(xlabel="epoch", ylabel="loss", legend=['train_loss', "test_loss"])
    for epoch in range(max_epochs):
        metric = d2l.Accumulator(4)
        iter = 1
        for X, Y in train_iter:
            X = X.to(device)
            Y = Y.to(device)
            output = net(X).squeeze(-1)
            loss = loss_fn(output, Y).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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

        print('acc : %.4f' % (metric[3]/metric[2]))
        animator.add(epoch+1, [metric[1]/metric[0], metric[3]/metric[2]])
    torch.save(net.state_dict(), "../models/params.pkl")
    plt.savefig(fname="../models/loss.jpg")


trainer(net, train_iter, test_iter, loss_fn, optimizer, max_epochs=max_epochs, device=device)

