import pandas as pd
from pandas import DataFrame
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import pickle


def load_time_data(steps: list, batch_size=32, time_steps=15, jump=False):
    data = pd.read_csv('../../data/202309221011205597/train_data.csv')
    n_train = len(data)
    pre_data = pd.read_csv('../../data/202309221011205597/dev_data.csv')
    data = pd.concat([data, pre_data], axis=0)    # 将预测数据和训练数据合并，从而进行归一化
    data.fillna(0, inplace=True)
    data.reset_index(inplace=True)
    data.drop(columns=['index', '大气压', '风速', '风向', '平均风速', '平均风向', '阵风速', '阵风向', '降雨量', '辐照度_POA', '无功功率',
                       '累计发电量'], inplace=True)

    # 去除异常值
    data = handling_outliers(data)
    # 增加时间特征
    data['time'] = data['时间'].apply(lambda x: pd.Timestamp(x))
    data['月'] = data['time'].apply(lambda x: x.month)
    data['日'] = data['time'].apply(lambda x: x.day)
    data['时'] = data['time'].apply(lambda x: x.hour)
    # data['分'] = data['time'].apply(lambda x: x.minute)
    data['一天中的第几分钟'] = data['time'].apply(lambda x: x.dayofyear)
    # 一年中的哪个季度
    season_dict = {
        1: 1, 2: 1, 3: 1,
        4: 2, 5: 2, 6: 2,
        7: 3, 8: 3, 9: 3,
        10: 4, 11: 4, 12: 4,
    }
    data['季节'] = data['月'].map(season_dict)

    data.drop(columns=['时间', '当日发电量', 'time'], inplace=True)
    data: DataFrame

    data, label_names = create_lable(data, steps, n_train)    # 创建label
    max_step = max(steps)
    data.drop(columns=['有功功率'], inplace=True)    # 创建完后把有功功率删除

    col_list = data.columns.tolist()
    for label_name in label_names:
        col_list.remove(label_name)       # 不对label列进行归一化
    data = norm_data(data, col_list)    # 对其他列进行归一化

    # 训练集得到时间步
    train_data = data[:n_train-max_step]
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
    valid_data = data.drop(columns=label_names)[n_train:]  # 将label列删除，生成验证集
    valid_data = np.array(valid_data)
    valid_data = get_time_steps(valid_data, num_steps=time_steps, jump=True)
    valid_data = torch.tensor(valid_data, dtype=torch.float32)
    with open("valid_data.pkl", "wb") as f:
        pickle.dump(valid_data, f)
    del valid_data


def norm_data(data, colunms: list):
    for colunm in colunms:
        data[colunm] = (data[colunm]-data[colunm].min()) / (data[colunm].max() - data[colunm].min()+1e-6)
    return data


def split_train_test(x_data, t_data, seed, rate):
    """
    按照seed和rate来区分训练集和测试集
    :param x_data: 未打乱的特征数据，np.array数组
    :param t_data: 未打乱的标签数据, np.array数组
    :param seed: 种子
    :param rate: 划分比例，eg:0.9代表训练集占0.9
    :return: 返回打乱后的(x_train, t_train), (x_test, t_test)， 均为np.array数组
    """
    train_n = round(x_data.shape[0] * rate)
    x_train, t_train = x_data[:train_n], t_data[:train_n]
    x_test, t_test = x_data[train_n:], t_data[train_n:]
    shuffled_indices = np.arange(train_n)
    np.random.seed(seed)
    np.random.shuffle(shuffled_indices)
    x_train, t_train = x_train[shuffled_indices], t_train[shuffled_indices]
    return (x_train, t_train), (x_test, t_test)


def create_lable(data, steps: list, n_train):
    max_step = max(steps)
    label_names = [str(step)+'分有功功率' for step in steps]
    for i in range(n_train-max_step):
        for step, label_name in zip(steps, label_names):
            data.loc[i, label_name] = data.loc[i+step, '有功功率']
    return data, label_names


def get_time_steps(data, num_steps, jump: (bool, int)):
    """
    对时间序列按时间步生成一条条数据
    :param data: 未分割的时间序列数据,np.array数组
    :param num_steps: 时间步
    :param jump: 是否间隔排序, bool, int
    :return: 未打乱的data, np.array数组
    """
    if jump == True:
        steps = len(data) // num_steps
        data = [data[i*num_steps:(i+1)*num_steps] for i in range(steps)]
    elif type(jump) == int:
        n = data.shape[0]
        if n % jump == 0:
            data = [data[i: i + num_steps] for i in range(0, n - num_steps + jump, jump)]
        else:
            data = [data[i: i + num_steps] for i in range(0, n - num_steps, jump)]
        print("number %d has been cut as %d" % (n, len(data)))
    else:
        n = data.shape[0]
        data = [data[i:i+num_steps] for i in range(n-num_steps+1)]
        data = np.array(data)
        print("number %d has been cut as %d" % (n, len(data)))
    data = np.array(data)
    return data


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


def load_linear_data(steps: list, batch_size=32, time_steps=15, jump=False):
    data = pd.read_csv('../../data/202309221011205597/train_data.csv')
    n_train = len(data)
    pre_data = pd.read_csv('../../data/202309221011205597/dev_data.csv')
    data = pd.concat([data, pre_data], axis=0)    # 将预测数据和训练数据合并，从而进行归一化
    data.fillna(0, inplace=True)
    data.reset_index(inplace=True)
    data.drop(columns=['index', '大气压', '风速', '风向', '平均风速', '平均风向', '阵风速', '阵风向', '降雨量', '辐照度_POA', '无功功率',
                       '累计发电量'], inplace=True)

    data['time'] = data['时间'].apply(lambda x: pd.Timestamp(x))
    data['月'] = data['time'].apply(lambda x: x.month)
    data['日'] = data['time'].apply(lambda x: x.day)
    data['时'] = data['time'].apply(lambda x: x.hour)
    data['分'] = data['time'].apply(lambda x: x.minute)

    data.drop(columns=['时间', '当日发电量', 'time'], inplace=True)
    data: DataFrame

    data, label_names = create_lable(data, steps, n_train)    # 创建label
    max_step = max(steps)
    data.drop(columns=['有功功率'], inplace=True)    # 创建完后把有功功率删除

    col_list = data.columns.tolist()
    for label_name in label_names:
        col_list.remove(label_name)       # 不对label列进行归一化
    data = norm_data(data, col_list)    # 对其他列进行归一化

    # 划分训练集和验证集
    train_data = data[:n_train - max_step]
    valid_data = data.drop(columns=label_names)[n_train:]   # 将label列删除，生成验证集

    train_data = np.array(train_data)
    valid_data = torch.tensor(np.array(valid_data), dtype=torch.float32)

    # 划分训练集和测试集
    X_data = train_data[:, :-len(label_names)]
    Y_data = train_data[:, -len(label_names):]

    (x_train, t_train), (x_test, t_test) = split_train_test(X_data, Y_data, 2023, 0.8)

    train_datasets = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                   torch.tensor(t_train, dtype=torch.float32))
    test_datasets = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                   torch.tensor(t_test, dtype=torch.float32))

    train_iter = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    print(next(iter(train_iter))[0].shape)
    test_iter = DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

    with open("train_data.pkl", "wb") as f:
        pickle.dump((train_iter, test_iter), f)
    with open("valid_data.pkl", "wb") as f:
        pickle.dump(valid_data, f)

    return train_iter, test_iter, valid_data


# steps = [15, 30, 60, 240, 1440]
steps = [15, 30, 60]
load_time_data(steps=steps, batch_size=128, time_steps=60, jump=False)

