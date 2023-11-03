from torch import nn, optim
from net import linear_net, LSTMWithLinear, SelfAttention, Bertblock, Bert
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import pickle


device = d2l.try_gpu()

# 加载数据
with open("train_data.pkl", "rb") as f:
    train_iter, test_iter = pickle.load(f)

# net = linear_net
# net = LSTMWithLinear(27, 20, 32, 3, 3, dropout=0.1)
# net = SelfAttention(27, num_heads=3, dropout=0.2, key_size=27, value_size=27, output_features=3, hidden_size=32)
net = Bert(in_features=27, hidden_size=20, num_heads=4, dropout=0, out_features=3, num_layers=3, num_steps=60)
loss_fn = nn.MSELoss()
lr = 0.01
new_lr = 0.001
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

        if metric[3]/metric[2] < 4.5:
            lr = new_lr
    torch.save(net.state_dict(), "params.pkl")
    plt.savefig(fname="../models/loss.jpg")


print("\n", next(iter(train_iter))[0].shape)
trainer(net, train_iter, test_iter, loss_fn, optimizer, max_epochs=max_epochs, device=device, lr=lr, new_lr=new_lr)