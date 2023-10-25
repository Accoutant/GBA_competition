from torch import nn
import torch


linear_net = nn.Sequential(nn.Linear(27, 200),
                    nn.ReLU(),
                    nn.Linear(200, 3))


def rmse(output, target):
    n = output.shape[0]
    loss = torch.sqrt(torch.pow((output-target)/(target + 1e-6), 2).sum()/n)
    return loss



