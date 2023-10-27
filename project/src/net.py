from sklearn.linear_model import Ridge
import pickle
import pandas as pd

with open("train_data.pkl", "rb") as f:
    (x_train, t_train), (x_test, t_test), valid_data = pickle.load(f)

linear = Ridge()
linear.fit(x_train, t_train)
print("trian score:", linear.score(x_train, t_train))
print("test score:", linear.score(x_test, t_test))

predict = pd.DataFrame(linear.predict(valid_data), columns=["15分", "30分", "60分"])

with open("predict.pkl", "wb") as f:
    pickle.dump(predict, f)




