import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from sklearn.ensemble import RandomForestRegressor


def read_district(filename):
    cols = ['district_hash', 'district_id']
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
    
    
def read_order(filename):
    cols = ['order_id', 'driver_id', 'passenger_id', 'start_district_hash', 'dest_district_hash', 'price', 'time']
    df = pd.read_csv(filename, header=None, sep='\t', names=cols)
    return df    


def read_order_fp(filename):
    df = pd.read_csv(filename, header=None, sep='\t',
        usecols=[1,4,6], names=['driver_id','start_district_hash','time']
        #names=['order_id','driver_id','passenger_id','start_district_hash','dest_district_hash','price','time']
        )
    return df
    
    
def process_order(df):
    df["answered"] = df['driver_id'].notnull().astype(int)
    df["time"] = pd.to_datetime(df["time"])
    df["day"] = df["time"].dt.day
    df["slot"] = df["time"].dt.hour * 6 + df["time"].dt.minute // 10 + 1
    # drop useless columns
    #df.drop(["order_id"], axis = 1, inplace = True)
    #df.drop(["driver_id"], axis = 1, inplace = True)     
    df.drop(["passenger_id"], axis = 1, inplace = True)   
    df.drop(["dest_district_hash"], axis = 1, inplace = True)
    df.drop(["district_hash"], axis = 1, inplace = True)
    df.drop(["time"], axis = 1, inplace = True)   

    
def mape(y_true, y_pred):
    y_pred = y_pred[y_true > 0]
    y_true = y_true[y_true > 0]
    return np.mean(np.abs((y_true - y_pred) / y_true))


#%%
path = "..\\citydata\\season_1"
train_path = path + "\\training_data"
test_path = path + "\\test_set_1"
order_path = "\\order_data\\order_data_2016-01-"

train_data = read_order(train_path + order_path + "21")
test_data = read_order(test_path + order_path + "22_test")
df_district = read_district(test_path + "\\cluster_map\\cluster_map")
print("df_district = read_district(...) -- DONE")

#%%
df0 = pd.concat([train_data, test_data])
df0 = pd.merge(df0, df_district, how="inner", left_on=['start_district_hash'], right_on=['district_hash'])
df0.drop(["start_district_hash"], axis = 1, inplace = True)
df1 = df0.copy()
process_order(df1)
print("process_order(df1) -- DONE")


#%%
#tmp = df1.groupby(['driver_id', 'district_id', 'day'])['answered'].agg({'count':'count'}).reset_index()
tmp = df1.groupby(['order_id'])['answered'].agg({'count':'count'}).reset_index()
tmp.sort_values(['count'], inplace=True)
print(tmp)  # one order could have multiple answered


"""
df2 = df1.groupby(['district_id', 'day', 'slot'])['answered'].agg({'request':'count', 'answer':'sum'}).reset_index()
no_data = pd.DataFrame(list(product(list(range(1,67)),list(range(1,145)))), columns=['id', 'slot'])
no_data["day"] = 21
no_data["answer"] = 0
no_data["request"] = 0
# df3 = pd.concat([df2, no_data]).drop_duplicates(subset=['id', 'day', 'slot'], keep='first')
df3 = pd.concat([df2, no_data]).drop_duplicates(subset=['district_id', 'day', 'slot'])
df3["gap"] = df3["request"] - df3["answer"]
df3.sort_values(['district_id','day','slot'], inplace=True)


#%%
df4 = df3[(df3["district_id"]==3)&(df3["day"]==21)]
#plt.plot(df4["slot"], df4["request"])
plt.plot(df4["slot"], df4["gap"])
plt.show()


#%%
df_train = df3[(df3["day"]==21)]
df_test = df3[(df3["day"]==22)]
reg_req = RandomForestRegressor(random_state = 0)
reg_req.fit(df_train[['district_id', 'slot']], df_train['request'])
predict_req = reg_req.predict(df_test[['district_id', 'slot']])
reg_ans = RandomForestRegressor(random_state = 0)
reg_ans.fit(df_train[['district_id', 'slot']], df_train['answer'])
predict_ans = reg_ans.predict(df_test[['district_id', 'slot']])
predict_gap = predict_req - predict_ans
predict_gap[predict_gap < 0] = 0
#df_test.insert(0, "predict_gap", predict_gap)
error = mape(df_test["gap"].values, predict_gap)
print(error)
"""