from net import linear_net, LSTMWithLinear
import torch
import pickle
import pandas as pd


net = LSTMWithLinear(26, 150, 1, 3)
# net = linear_net
net.load_state_dict(torch.load('params.pkl'))
with open("valid_data.pkl", "rb") as f:
    valid_data = pickle.load(f)


def predict(net, valid_data):
    output = net(valid_data)
    output = pd.DataFrame(output.detach().numpy(), columns=["15分", "30分", "60分", "240分", "1440分"])

    valid = pd.read_csv('../../data/202309221011205597/result_sample.csv')
    valid['15min'] = output['15分']
    valid['30min'] = output['30分']
    valid['1h'] = output['60分']
    #valid['4h'] = output['240分']
    #valid['24h'] = output['1440分']
    valid.to_csv("../models/predict.csv", index=None)


def predict_lstm(net, valid_data):
    output = net(valid_data)
    output = torch.flatten(output, start_dim=0, end_dim=1)
    output = pd.DataFrame(output.detach().numpy(), columns=["15分", "30分", "60分"])

    valid = pd.read_csv('../../data/202309221011205597/result_sample.csv')
    valid['15min'] = output['15分']
    valid['30min'] = output['30分']
    valid['1h'] = output['60分']
    # valid['4h'] = output['240分']
    # valid['24h'] = output['1440分']
    valid.to_csv("../models/predict.csv", index=None)


predict_lstm(net, valid_data)


