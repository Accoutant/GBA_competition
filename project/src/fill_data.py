import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_kmo, FactorAnalyzer
from process_data import norm_data, create_lable, get_time_steps, split_train_test
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
import torch
import pickle


def factory_analysis(data, label_names, date_labels, n_factor):
    data.drop(columns=['index', '大气压', '降雨量', '有功功率', "累计发电量", "当日发电量"] + label_names, inplace=True)
    columns = data.columns.tolist()
    data = norm_data(data, columns)
    print("data_shape:", data.shape)
    print("columns:", columns)
    kmo_all, kmo_model = calculate_kmo(data)
    print(kmo_all, kmo_model)

    faa = FactorAnalyzer(data.shape[1], rotation=None)
    faa.fit(data)
    ev, v = faa.get_eigenvalues()
    print("特征值和特征向量:", ev, v)
    # 可视化展示
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.title("Scree Plot")
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalue")
    sns.lineplot(ev, markers="o")
    sns.scatterplot(ev)
    plt.savefig("faa.jpg")
    plt.plot()

    # 确认因子个数后，分析因子间相关性
    faa_six = FactorAnalyzer(n_factor, rotation="varimax")
    faa_six.fit(data)
    var = pd.DataFrame(faa_six.get_factor_variance())
    print("factor_var:", var)

    # 热力图
    heat = pd.DataFrame(np.abs(faa_six.loadings_).T, columns=columns)
    plt.close()
    plt.figure(figsize=(40, 10))
    sns.heatmap(heat, cmap='BuPu')
    plt.savefig("factor_heat.jpg")
    # plt.show()

    # 转换为新的变量
    data_new = faa_six.transform(data)
    data_new = pd.DataFrame(data_new)
    data_new = pd.concat([data_new, date_labels], axis=1)
    return data_new


def creat_time_feature(date, data, label_names):
    date['time'] = date['时间'].apply(lambda x: pd.Timestamp(x))
    date['月'] = date['time'].apply(lambda x: x.month)
    date['日'] = date['time'].apply(lambda x: x.day)
    date['时'] = date['time'].apply(lambda x: x.hour)
    # date['分'] = date['time'].apply(lambda x: x.minute)
    date['一天中的第几分钟'] = date['time'].apply(lambda x: x.dayofyear)
    # 一年中的哪个季度
    season_dict = {
        1: 1, 2: 1, 3: 1,
        4: 2, 5: 2, 6: 2,
        7: 3, 8: 3, 9: 3,
        10: 4, 11: 4, 12: 4,
    }
    date['季节'] = date['月'].map(season_dict)
    date.drop(columns=['时间', 'time'], inplace=True)
    date = norm_data(date, colunms=date.columns.tolist())
    date = pd.concat([date, data[label_names]], axis=1)
    return date


def get_factors(steps: list, n_factor=6, time_steps=60, jump=False, batch_size=128):
    data = pd.read_csv('../../data/202309221011205597/train_data.csv')
    n_train = len(data)
    pre_data = pd.read_csv('../../data/202309221011205597/dev_data.csv')
    data = pd.concat([data, pre_data], axis=0)    # 将预测数据和训练数据合并，从而进行归一化
    data.reset_index(inplace=True)

    # 创造label
    data, label_names = create_lable(data, steps, n_train)

    # 独立出时间特征和labels
    date_labels = pd.DataFrame(data['时间'])
    data.drop(columns=['时间'], inplace=True)
    date_labels = creat_time_feature(date_labels, data, label_names)

    # 开始因子分析-
    data_new = factory_analysis(data, label_names, date_labels, n_factor)
    
    # 建造数据集和验证集
    max_step = max(steps)
    train_data = data_new[:n_train - max_step]
    train_data = np.array(train_data)
    train_data = get_time_steps(train_data, num_steps=time_steps, jump=jump)

    # 划分训练集和测试集
    X_data = train_data[:, :, :-len(label_names)]
    Y_data = train_data[:, :, -len(label_names):]
    del train_data
    (x_train, t_train), (x_test, t_test) = split_train_test(X_data, Y_data, 2023, 0.8)
    del X_data, Y_data
    # 训练集得到datasets
    train_datasets = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                   torch.tensor(t_train, dtype=torch.float32))
    test_datasets = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                  torch.tensor(t_test, dtype=torch.float32))

    train_iter = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_datasets, batch_size=batch_size, shuffle=True)
    print("\n", next(iter(train_iter))[0].shape)
    with open("train_data.pkl", "wb") as f:
        pickle.dump((train_iter, test_iter), f)
    del train_datasets, test_datasets, train_iter, test_iter

    # 测试集得到时间步
    valid_data = data_new.drop(columns=label_names)[n_train:]  # 将label列删除，生成验证集
    valid_data = np.array(valid_data)
    valid_data = get_time_steps(valid_data, num_steps=time_steps, jump=True)
    valid_data = torch.tensor(valid_data, dtype=torch.float32)
    with open("valid_data.pkl", "wb") as f:
        pickle.dump(valid_data, f)
    del valid_data

steps = [15, 30, 60]
get_factors(steps, n_factor=6, time_steps=60, jump=2, batch_size=128)


