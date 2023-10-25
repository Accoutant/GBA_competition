from net import linear_net
import torch
import pickle
import pandas as pd

net = linear_net
net.load_state_dict(torch.load('../models/params.pkl'))
with open("valid_data.pkl", "rb") as f:
    valid_data = pickle.load(f)


def predict(net, valid_data):
    output = net(valid_data)
    output = pd.DataFrame(output.detach().numpy())

    output.to_excel("../models/predict.xlsx")


predict(net, valid_data)


