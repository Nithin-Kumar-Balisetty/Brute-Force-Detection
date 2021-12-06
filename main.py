import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split


def helper(x, y):
    tmp_model = LinearRegression()
    tmp_model.fit(y.reshape(-1, 1), x.reshape(-1, 1))
    return tmp_model.coef_.tolist()[0]


def testing():
    print("hello")
    table = pd.read_csv("./CSV.csv")
    filter_table = table.groupby("Source")
    keys = (filter_table.groups.keys())
    for i in keys:
        print(filter_table.get_group(i))


def RModel():
    main_table = pd.read_csv("./CSV.csv")
    filter_table = main_table.groupby("Source")
    keys = filter_table.groups.keys()
    for key in keys:
        table = filter_table.get_group(key)
        time = table["Time"].head(50).to_numpy()
        pkno = [(i - 1) for i in table["No."].head(50).to_numpy()]
        slope = []

        for i in range(0, len(table.index)):
            slope.append(helper(time[0: i + 1], np.array(pkno[0: i + 1])))

        y_array = np.add(np.sqrt(table['Time Diff'].head(50).to_numpy()), (table['Time'].head(50).to_numpy()))
        # y_array = (table['Time'].head(20).to_numpy())
        # y_array = np.multiply(y_array, (np.array(slope)))
        x_array = table["No."].head(50).to_numpy()

        # x_array = np.array(pkno)

        x = np.array(x_array).reshape((-1, 1))
        y = np.array(y_array)

        print(x)
        print(len(y))
        model = LinearRegression()
        model.fit(x, y)
        r_sq = model.score(x, y)

        print('coefficient of determination:', r_sq)

        plt.plot(x_array, y_array, color='b')

        plt.xlabel('packet number')
        plt.ylabel('Time diff')

        y_pred = model.predict(x)

        # Linear Eqn
        plt.plot(x, y_pred, color='k')
        plt.show()


if __name__ == '__main__':
    print("....Starting....")
    RModel()
    # testing()
