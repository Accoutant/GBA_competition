import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def handling_outliers(data):
    data_zscore = data.copy()
    cols = data.columns
    for col in cols:
        if data[col].dtype != "object":
            data_col = data[col]
            z_score = (data_col - data_col.mean()) / (data_col.std() + 1e-5)
            data_zscore[col] = z_score.abs() > 3
        else:
            data_zscore[col] = False
    data[data_zscore] = np.nan
    print(data.isnull().sum())
    data.interpolate(inplace=True)
    return data


data = pd.read_csv('../../data/202309221011205597/train_data.csv')
print(data.describe())

data.drop(columns=['累计发电量', "时间"], inplace=True)
plt.figure(figsize=(40, 20))
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

data.boxplot()
plt.savefig("./box.jpg")

data = handling_outliers(data)
print(data)

