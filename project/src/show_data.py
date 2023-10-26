import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from process_data import create_lable

data = pd.read_csv('../../data/202309221011205597/train_data.csv')


def show_heatmap(data):
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

    data, label_names = create_lable(data, [15, 30, 60], len(data))   # 创造标签
    # data.drop(columns=['时间'], inplace=True)
    data.fillna(0, inplace=True)
    corr = data.corr()
    corr.to_excel('../models/coor.xlsx')

    corr.fillna(0, inplace=True)
    sns.set(font='SimHei')
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr, cmap='OrRd')
    plt.savefig('../models/heatmap.jpg')
    plt.clf()


def show_profile(data: DataFrame):
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

    data = data[['时间', '有功功率']]
    sns.set(font='SimHei')
    sns.histplot(data)
    plt.savefig('../models/profilemap.jpg')
    plt.clf()


def show_timemap(data: DataFrame):
    data = data[['时间', '有功功率']]
    sns.lineplot(data)
    plt.savefig('../models/timemap.jpg')
    plt.clf()


show_heatmap(data)
show_profile(data)
show_timemap(data)

