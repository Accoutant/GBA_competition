from torch import nn
import torch
from d2l import torch as d2l


linear_net = nn.Sequential(nn.Linear(26, 100),
                           nn.ReLU(),
                           nn.Linear(100, 100),
                           nn.ReLU(),
                           nn.Linear(100, 3))


class LSTMWithLinear(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, out_features, dropout=0, bidirectional=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(input_size, hidden_size1), nn.ReLU())
        self.lstm = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.linear2 = nn.Linear(hidden_size2, out_features)

    def forward(self, X):
        # X.shape: batch_size, num_steps, input_size
        X = self.linear1(X)
        Y, _ = self.lstm(X)
        output= self.linear2(Y)
        return output

    def init_weight(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)


def rmse(output, target):
    n = output.shape[0]
    loss = torch.sqrt(torch.pow((output-target)/(target + 1e-6), 2).sum()/n)
    return loss


class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, key_size, value_size, output_features, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout,
                                               kdim=key_size, vdim=value_size)
        self.linear = nn.Sequential(nn.Linear(embed_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, output_features))

    def forward(self, X):
        # X.shape: batch_size, num_steps, num_features
        X = X.permute(1, 0, 2)
        attention_output, attention_weight = self.attention(X, X, X)
        attention_output = attention_output.permute(1, 0, 2)
        output = self.linear(attention_output)
        return output


# net = SelfAttention(27, 3, dropout=0.1, key_size=27, value_size=27, output_features=3)


class Bertblock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, kdim=hidden_size, vdim=hidden_size)
        self.addnorm1 = d2l.AddNorm(hidden_size, dropout)
        self.feed = nn.Linear(hidden_size, hidden_size)
        self.addnorm2 = AddNorm(60, dropout=dropout)

    def forward(self, X):
        # X.shape: batch_size, num_steps, num_features
        X = X.permute(1, 0, 2)
        attention_output, attention_weight = self.attention(X, X, X)
        attention_output = attention_output.permute(1, 0, 2)
        Y1 = self.addnorm1(X.permute(1, 0, 2), attention_output)
        Y2 = self.feed(Y1)
        Y3 = self.addnorm2(Y1, Y2)
        return Y3


class Bert(nn.Module):
    def __init__(self, in_features, hidden_size, num_heads, dropout, out_features, num_layers):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module("bertblock"+str(i), Bertblock(hidden_size, num_heads, dropout))
        self.linear2 = nn.Linear(hidden_size, out_features)

    def forward(self, X):
        X1 = self.linear1(X)
        X2 = self.blocks(X1)
        X3 = self.linear2(X2)
        return X3


class AddNorm(nn.Module):
    """Residual connection followed by layer normalization.

    Defined in :numref:`sec_transformer`"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.BatchNorm1d(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)