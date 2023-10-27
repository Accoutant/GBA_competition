from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
import pickle
import pandas as pd
import numpy as np


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


labels = ["15分", "30分", "60分"]
predict = use_svm(labels)




