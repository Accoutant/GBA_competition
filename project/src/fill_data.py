import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('../../data/202309221011205597/train_data.csv')
print(data.describe())

data.drop(columns=['累计发电量', "时间"], inplace=True)
plt.figure(figsize=(40, 20))
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

data.boxplot()
plt.savefig("./box.jpg")

