import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from sklearn.ensemble import RandomForestRegressor

def read_district(filename):
    cols = ['district_hash', 'district_id']
    df = pd.read_csv(filename, header=None, sep='\t', names=cols)
    return df

def read_order(filename):
    cols = ['order_id', 'driver_id', 'passenger_id', 'start_district_hash', 'dest_district_hash', 'price', 'time']
    df = pd.read_csv(filename, header=None, sep='\t', names=cols)
    return df

def read_weather(filename):
    cols = ['Time', 'Weather', 'temperature', 'PM2.5']
    df = pd.read_csv(filename, header=None, sep='\t', names=cols)
    return df
    
    
def read_traffic(filename):
    cols = ['district_hash', 'tj_level', 'tj_time']
    df = pd.read_csv(filename, header=None, sep='\t', names=cols)
    return df    
    
def read_poi(filename):
    cols = ['district_hash', 'poi_class']
    df = pd.read_csv(filename, header=None, sep='\t', names=cols)
    return df
    
def read_submit(filename): 
    cols = ['year','month','day','slot']
    df = pd.read_csv(filename, header=0, sep='-', names=cols)
    df['time'] = pd.read_csv(filename, header=0, names=['time'])    
    return df
   
def process_order(df):
    df["answered"] = df['driver_id'].notnull().astype(int)
    df["time"] = pd.to_datetime(df["time"])
    df["day"] = df["time"].dt.day
    df["slot"] = df["time"].dt.hour * 6 + df["time"].dt.minute // 10 + 1
    
    cols = ["order_id","driver_id","passenger_id","dest_district_hash","time","price"]
    df.drop(cols, axis = 1, inplace = True)
    
def mape(y_true, y_pred):
    y_pred = y_pred[y_true > 0]
    y_true = y_true[y_true > 0]
    return np.mean(np.abs((y_true - y_pred) / y_true))

#%%
if __name__ == '__main__':
    path = ".\\citydata\\season_1"
    train_path = path + "\\training_data"
    test_path = path + "\\test_set_1"
    
    order_path = "\\order_data\\order_data_2016-01-"    
    train_data = pd.concat(read_order(train_path + order_path + str(i).zfill(2)) for i in range(1, 22))
    test_data = pd.concat(read_order(test_path + order_path + str(i).zfill(2) + "_test") for i in [22,24,26,28,30])
    submit_data = read_submit(test_path + "\\read_me_1.txt")
    
    id_data = read_district(test_path + "\\cluster_map\\cluster_map")

    #%%    
    df1 = pd.concat([train_data, test_data])
    df1 = pd.merge(df1, id_data, left_on=['start_district_hash'], right_on=['district_hash'])
    df1.drop(["start_district_hash", "district_hash"], axis = 1, inplace = True)    
    process_order(df1)

    #%%
    df2 = df1.groupby(['district_id', 'day', 'slot'])['answered'].agg({'request':'count', 'answer':'sum'}).reset_index()
    no_data = pd.DataFrame(list(product(list(range(1,67)),list(range(1,145)))), columns=['district_id', 'slot'])
    no_data["day"] = 21
    no_data["answer"] = 0
    no_data["request"] = 0
    df3 = pd.concat([df2, no_data]).drop_duplicates(subset=['district_id', 'day', 'slot'], keep='first')
    df3["gap"] = df3["request"] - df3["answer"]
    df3.sort_values(['district_id','day','slot'], inplace=True)
    
    #%%
    df4 = df3[(df3["district_id"]==3)&(df3["day"]==21)]
    #plt.plot(df4["slot"], df4["request"])
    plt.plot(df4["slot"], df4["gap"])
    plt.show()
    
    #%%
    df_train = df3[(df3["day"]<=21)]
    df_test = df3[(df3["day"]>=22)]
    cols = ['district_id','slot']
    reg_req = RandomForestRegressor(random_state = 0)
    reg_req.fit(df_train[cols], df_train['request'])
    predict_req = reg_req.predict(df_test[cols])
    reg_ans = RandomForestRegressor(random_state = 0)
    reg_ans.fit(df_train[cols], df_train['answer'])
    predict_ans = reg_ans.predict(df_test[cols])
    predict_gap = predict_req - predict_ans
    predict_gap[predict_gap < 0] = 0
    #df_test.insert(0, "predict_gap", predict_gap)
    error = mape(df_test["gap"].values, predict_gap)
    print(error)
    
    #%%
    df_submit = pd.DataFrame(list(range(1,67)),columns=['district_id'])
    df_submit["key"] = 0
    submit_data["key"] = 0
    df_submit = pd.merge(df_submit, submit_data, how='outer', on='key')
    df_submit.drop(['key'], axis = 1, inplace = True)
    predict_req = reg_req.predict(df_submit[cols])
    predict_ans = reg_ans.predict(df_submit[cols])
    predict_gap = predict_req - predict_ans
    predict_gap[predict_gap < 0] = 0
    df_submit['gap'] = predict_gap
    df_submit.to_csv('submit.csv', header=False, index=False, columns=['district_id','time','gap'])