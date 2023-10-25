from torch import nn, optim
import process_data
from net import net, rmse
from torch.utils.tensorboard import SummaryWriter
import torch
from d2l import torch as d2l

steps = [15, 30, 60]
device = d2l.try_gpu()
train_iter, test_iter, valid_data = process_data.load_data(steps=steps, batch_size=256)
net = net
loss_fn = nn.MSELoss()
lr = 0.001
max_epochs = 25
optimizer = optim.SGD(net.parameters(), lr=lr)
writer = SummaryWriter('../logs')


def trainer(net, train_iter, test_iter, loss_fn, optimizer, max_epochs, device):
    net = net.to(device)
    for epoch in range(max_epochs):
        iter = 1
        for X, Y in train_iter:
            X = X.to(device)
            Y = Y.to(device)
            output = net(X).squeeze(-1)
            loss = loss_fn(output, Y).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('| epoch %d | iter %d | loss %.4f |' % (epoch+1, iter, loss))
            writer.add_scalar("loss", loss, iter)
            iter += 1

        metric = d2l.Accumulator(2)
        metric.reset()
        for X_test, Y_test in test_iter:
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)
            with torch.no_grad():
                output_test = net(X_test).squeeze(-1)
                loss_test = loss_fn(output_test, Y_test)
                metric.add(1, loss_test)

        print('acc : %.4f' % (metric[1]/metric[0]))
        writer.add_scalar("test_acc", metric[1]/metric[0], epoch+1)
    torch.save(net.state_dict(), "../models/params.pkl")


trainer(net, train_iter, test_iter, loss_fn, optimizer, max_epochs=max_epochs, device=device)

