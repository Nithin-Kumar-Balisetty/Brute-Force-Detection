import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import random as rn

def helper(x, y):
    tmp_model = LinearRegression()
    tmp_model.fit(y.reshape(-1, 1), x.reshape(-1, 1))
    return tmp_model.coef_.tolist()[0]


def testing(table_name):
    print("hello")
    table = pd.read_csv(table_name)
    filter_table = table.groupby("Source")
    keys = (filter_table.groups.keys())
    for i in keys:
        f_table = filter_table.get_group(i)
        print(f_table.loc[f_table['Info'].str.contains('GET /CSIS/', case=False)])


def RModel(table_name,size ,route):
    main_table = pd.read_csv(table_name)
    filter_table = main_table.groupby("Source")
    keys = filter_table.groups.keys()
    for key in keys:
        before_tab = filter_table.get_group(key)
        table = before_tab.loc[before_tab['Info'].str.contains(route, case=False)]

        if len(table["Source"].to_numpy()) < size:
            continue
        l= len(table["Source"].to_numpy())
        time_array = table["Time"].head(l).to_numpy()
        main_time = [0.000000]
        for i in range(1, l):
            #main_time.append(rn.randint(main_time[i-1], 100))
            main_time.append(time_array[i] - time_array[0])

        time_diff = [main_time[1]]
        for i in range(1, l):
            time_diff.append(main_time[i]-main_time[i-1])



        #y_array = np.add(np.sqrt(table['Time Diff'].head(size).to_numpy()), (table['Time'].head(size).to_numpy()))
        #y_array = np.add(np.sqrt((np.array(time_diff))), (table['Time'].head(size).to_numpy()))
        y_array = np.array(main_time)
        #y_array = np.add(np.sqrt((np.array(time_diff))), (np.array(main_time)))
        print(y_array)
        temp = []
        for i in range(1, l+1):
            temp.append(i)
        x_array = np.asarray(temp)

        x = np.array(x_array).reshape(-1, 1)
        y = np.array(y_array).reshape(-1, 1)

        model = LinearRegression()
        model.fit(x, y)
        r_sq = model.score(x, y)

        print('coefficient of determination:', r_sq)
        print("coefficient of correlation: ", pearsonr(main_time, temp))
        plt.plot(x_array, y_array, color='b')

        plt.xlabel('Packet Number (Number of Packets)')
        plt.ylabel('Time')

        y_pred = model.predict(x)

        # Linear Eqn
        plt.plot(x, y_pred, color='k')
        plt.show()

RModel("./100ReqBruteForce.csv", 50 , "GET /otaku/anime")

#testing("./bigFlows.csv")
