from torch import nn
import torch


linear_net = nn.Sequential(nn.Linear(26, 100),
                           nn.ReLU(),
                           nn.Linear(100, 100),
                           nn.ReLU(),
                           nn.Linear(100, 3))


class LSTMWithLinear(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, out_features, dropout=0):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(input_size, hidden_size1), nn.ReLU())
        self.lstms = nn.ModuleList()
        for i in range(out_features):
            self.lstms.add_module("lstm"+str(i), nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2,
                                                         num_layers=num_layers, batch_first=True, dropout=dropout))
        self.affine = nn.Linear(hidden_size2, 1)
        self.affines = nn.ModuleList()
        for i in range(out_features):
            self.affines.add_module("affine"+str(i), self.affine)

        # 将affine层初始化参数
        for affine in self.affines:
            affine.apply(self.init_weight)

    def forward(self, X):
        X = self.linear(X)
        lstm_outputs = []
        for lstm in self.lstms:
            X1, _ = lstm(X)
            lstm_outputs.append(X1)
        affine_outputs = []
        for X1, affine in zip(lstm_outputs, self.affines):
            X2 = affine(X1)
            affine_outputs.append(X2)
        return torch.cat(affine_outputs, dim=-1)

    def init_weight(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)


def rmse(output, target):
    n = output.shape[0]
    loss = torch.sqrt(torch.pow((output-target)/(target + 1e-6), 2).sum()/n)
    return loss


class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, key_size, value_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout,
                                               kdim=key_size, vdim=value_size, batch_first=True)
        self.linear = nn.Linear(embed_size, 1)

    def forward(self, X):
        # X.shape: batch_size, num_steps, num_features
        attention_output, attention_weight = self.attention(X, X, X)
        output = self.linear(attention_output)
        return output.squeeze(-1)


# net = SelfAttention(27, 3, dropout=0.1, key_size=27, value_size=27)



