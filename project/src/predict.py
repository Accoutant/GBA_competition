import pickle
import pandas as pd

with open("predict.pkl", "rb") as f:
    data = pickle.load(f)


def predict(data):
    valid = pd.read_csv('../../data/202309221011205597/result_sample.csv')
    valid['15min'] = data['15分']
    valid['30min'] = data['30分']
    valid['1h'] = data['60分']
    #valid['4h'] = data['240分']
    #valid['24h'] = data['1440分']
    valid.to_csv("../models/predict.csv", index=None)


predict(data)


