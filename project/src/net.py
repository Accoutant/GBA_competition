from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def use_linear(labels: list):
    with open("train_data.pkl", "rb") as f:
        (x_train, t_train), (x_test, t_test), valid_data = pickle.load(f)

    linear = LinearRegression()
    linear.fit(x_train, t_train)
    print("trian score:", linear.score(x_train, t_train))
    print("test score:", linear.score(x_test, t_test))

    predict = pd.DataFrame(linear.predict(valid_data), columns=labels)

    with open("predict.pkl", "wb") as f:
        pickle.dump(predict, f)


def use_svm(labels: list):
    svms = [LinearSVR() for _ in range(len(labels))]
    with open("train_data.pkl", "rb") as f:
        (x_train, t_train), (x_test, t_test), valid_data = pickle.load(f)

    outputs = []
    for i, svm in enumerate(svms):
        svm = LinearSVR()
        svm.fit(x_train, t_train[:, i])
        print("net_train", i, ":", svm.score(x_train, t_train[:, i]))
        print("net_test", i, ":", svm.score(x_test, t_test[:, i]))
        output = svm.predict(valid_data)
        output = np.expand_dims(output, axis=-1)
        outputs.append(output)

    predict = pd.DataFrame(np.concatenate(outputs, axis=1), columns=labels)

    with open("predict.pkl", "wb") as f:
        pickle.dump(predict, f)

    return predict


def use_pca():
    with open("train_data.pkl", "rb") as f:
        (x_train, t_train), (x_test, t_test), valid_data = pickle.load(f)

    pca = PCA(n_components=1)
    pca.fit(x_train)
    x_pca = pca.transform(x_train)
    print(pca.explained_variance_)
    print(pca.score(x_test,t_test))
    print(x_pca.shape)


def use_xgboost():
    with open("train_data.pkl", "rb") as f:
        (x_train, t_train), (x_test, t_test), valid_data = pickle.load(f)
    dtrain = xgb.DMatrix(x_train, label=t_train)
    dtest = xgb.DMatrix(x_test, label=t_test)
    params = {
        'max_depth': 4,  # 每棵决策树的最大深度
        'eta': 0.1,  # 学习率
        'subsample': 0.7,  # 每次随机选择的样本比例
        'colsample_bytree': 0.7,  # 每棵决策树随机选择的特征比例
        'objective': 'reg:squarederror',  # 损失函数
        'eval_metric': 'rmse',  # 评价指标
        'silent': 1  # 是否输出日志信息
    }
    # 训练 XGBoost 模型
    num_round = 100  # 决策树的数量
    bst = xgb.train(params, dtrain, num_round)

    # 使用测试集进行预测
    t_pred = bst.predict(dtest)
    print('RMSE:', mean_squared_error(t_test, t_pred, squared=False))
    predict = bst.predict(xgb.DMatrix(valid_data))
    predict = pd.DataFrame(predict, columns=['15分', '30分', '60分'])
    with open("predict.pkl", "wb") as f:
        pickle.dump(predict, f)

# labels = ["15分", "30分", "60分"]
# predict = use_svm(labels)
# use_pca()
use_xgboost()


