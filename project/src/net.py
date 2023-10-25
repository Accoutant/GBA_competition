from torch import nn
import torch


net = nn.Sequential(nn.Linear(27, 100),
                    nn.ReLU(),
                    nn.Linear(100, 3))


def rmse(output, target):
    n = output.shape[0]
    loss = torch.sqrt(torch.pow((output-target)/(target + 1e-6), 2).sum()/n)
    return loss


def get_time_steps(data, num_steps, jump=True):
    """
    对时间序列按时间步生成一条条数据
    :param data: 未分割的时间序列数据,np.array数组
    :param num_steps: 时间步
    :param jump: 是否间隔排序
    :return: 未打乱的data, np.array数组
    """
    if jump:
        steps = len(data) // num_steps
        data = [data[i*num_steps:(i+1)*num_steps] for i in range(steps)]
        data = np.array(data)
    else:
        n = data.shape[0]
        data = [data[i:i+num_steps] for i in range(n-num_steps+1)]
        data = np.array(data)
    return data
