import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from copy import deepcopy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

#file location of dataset
filename = "/Users/tina/Downloads/Bio_Conscious_Data_Challenge/"

df1 = pd.read_csv(filename+"heart_rate_data.csv")
df1 = df1.drop("timezone_offset", axis=1) # Drop unwanted column
print(df1.head())

df2 = pd.read_csv(filename+"distance_activity_data.csv")
df2 = df2.drop("timezone_offset", axis=1)
df2 = df2.drop("device", axis=1)
print(df2.head())

df3 = pd.read_csv(filename+"blood_glucose_data.csv")
df3 = df3.drop("timezone_offset", axis=1)
df3 = df3.sort_values(by=['point_timestamp'])
print(df3.head())

# round the min and drop sec to find more match between 3 data
def rounding_time(df):
    Time = (df['point_timestamp'].str.split(" ",expand=True).iloc[:,1])
    Date = (df['point_timestamp'].str.split(" ",expand=True).iloc[:,0])
    scater = Time.str.split(":",expand=True).astype(int)
    scater.iloc[:,1] = (((scater.iloc[:,1]+round(scater.iloc[:,2])/60)//4.5)*5).astype(int)
    new_t = Date +' '+scater.iloc[:,0].astype("string").str.zfill(2) + ':'+scater.iloc[:,1].astype("string").str.zfill(2)
    return new_t

df1['new_date'] = rounding_time(df1)
df1.drop_duplicates(subset=['new_date'],inplace=True) #drop duplicate Time
df2['new_date'] = rounding_time(df2)
df2.drop_duplicates(subset=['new_date'],inplace=True)
df3['new_date'] = rounding_time(df3)

# merge 3 features based blood suger
merge1 = pd.merge(df3, df2[['point_value(kilometers)','new_date']], on=['new_date'], how='left')
merge2 = pd.merge(merge1, df1[['point_value','new_date']], on=['new_date'], how='left')
print('number of distance nan values: ',merge2['point_value(kilometers)'].isna().sum())
print('number of heart rate nan values: ',merge2['point_value'].isna().sum())
merge2.point_value.interpolate(inplace = True) #interpolate nan values
merge2['point_value(kilometers)'].interpolate(inplace = True)
merge2.dropna(inplace=True) # Drop all rows with missing values
print(merge2.head())

Date = pd.to_datetime(merge2['point_timestamp'].str.split(" ",expand=True).iloc[:,0])
train_index = (Date >= '2017-05-16') & (Date < '2017-06-01')
valid_index = (Date >= '2017-06-01') & (Date < '2017-06-15')
test_index = (Date >= '2017-06-15')

merge2['Date'] = Date
merge2['month'] = Date.dt.month; dataset_by_month = merge2.resample('M',on='Date').sum()

plt.figure();plt.plot(merge2.index,merge2['point_value(mg/dL)'])
plt.ylabel('blood glucose(mg/dL)');plt.xlabel('Time(day)')

def create_data(Data,timelap=60):
    X,Y = [], []
    for i in range(Data.shape[0]-timelap):
        X.append(Data[i:i+timelap])
        Y.append(Data[i+timelap])
    return np.array(X),np.array(Y)

lap=24
dataset = merge2[["point_value(kilometers)","point_value","point_value(mg/dL)"]].values

#normalize features by max min
scaler = preprocessing.MinMaxScaler().fit(dataset[train_index]) 
x_train, y_train = create_data(scaler.transform(dataset[train_index]),timelap=lap)
x_test, y_test = create_data(scaler.transform(dataset[test_index]),timelap=lap)
x_valid, y_valid = create_data(scaler.transform(dataset[valid_index]),timelap=lap)

#normalize time-series data by mean std
meanstd = False
if('meanstd'):
    scaler2 = preprocessing.StandardScaler().fit(x_train[:,:,0])
    x_train[:,:,0] = scaler2.transform(x_train[:,:,0])
    x_test[:,:,0] = scaler2.transform(x_test[:,:,0])
    x_valid[:,:,0] = scaler2.transform(x_valid[:,:,0])

def lstm(units,X):
    model = Sequential()
    model.add(LSTM(units = units, return_sequences=True, 
              input_shape=(X.shape[1], X.shape[2])))
    model.add((LSTM(units = units)))
    model.add(Dense(3))
    #Compiler
    optimizer = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=optimizer,loss='mse')
    return model

def gru(units,X):
    model = Sequential()
    # Input layer
    model.add(GRU(units = units, return_sequences = True, 
                   input_shape = [X.shape[1], X.shape[2]]))
    #model.add(GRU(units = units, return_sequences = True))
    #model.add(Dropout(0.5)) 
    # Hidden layer
    model.add(GRU(units = units)) 
    #model.add(Dropout(0.5))
    model.add(Dense(units = 3)) 
    #Compile model
    optimizer = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=optimizer,loss='mse')
    return model

if(False):
    Model = gru(3,x_train)
    model_name = 'GRU_model'
else:
    Model = lstm(3,x_train)
    model_name = 'LSTM_model'

early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                           patience = 15)
history = Model.fit(x_train, y_train, epochs = 200,  
                    validation_data=(x_valid, y_valid),
                    batch_size = 32, shuffle = False, 
                    callbacks = [early_stop])
plt.figure();
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(model_name+' train vs validation loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['Train loss', 'Validation loss'], loc='upper right')
#plt.savefig(filename+'/figures/'+model_name+'_5min_val_train_loss.pdf', format='pdf')

print(Model.summary())

pred = Model.predict(x_test)
# rescaling the data to original scale
Error = scaler.inverse_transform(pred)[:,2] - scaler.inverse_transform(y_test)[:,2]
mse = np.square(Error).mean()
rmse = np.sqrt(mse)
mae = np.abs(Error).mean()
print(model_name+' Mean Absolute Error: {:.3f}'.format(mae))
print(model_name+' Root Mean Square Error: {:.3f}'.format(rmse))

# 1 step prediction plot
plt.figure()
plt.plot(scaler.inverse_transform(pred)[:,2])
plt.plot(scaler.inverse_transform(y_test)[:,2])
plt.legend(['prediction','original'],loc='upper left')
#plt.savefig(filename+'/figures/'+model_name+'_5min_'+'5pred.pdf', format='pdf')

plt.figure()
plt.plot(scaler.inverse_transform(pred)[:36,2])
plt.plot(scaler.inverse_transform(y_test)[:36,2])
plt.legend(['prediction','original'],loc='upper left')
#plt.savefig(filename+'/figures/'+model_name+'_5min_first_three_hours_'+'5pred.pdf', format='pdf')

# 60 min prediction plot
norm = False
pred_60min = [] ; upper_limit = []; lower_limit = []
pred_len = 60//5
testing_data = scaler.transform(dataset[test_index])
index = 2
First_data = testing_data[-pred_len-x_test.shape[1]*(index):-pred_len-x_test.shape[1]*(index-1)].reshape(1,-1,3)
#First_data = testing_data[:x_test.shape[1]].reshape(1,-1,3)
#first method to find confidence interval range
"""
for i in range(pred_len):
    if(i==0):
        data = First_data
    new_pred = Model.predict(data)
    if(i==0):
        #calculating the confidence interval of the perdiction
        up_data = new_pred + 1.96*(data[:,-1]-new_pred) 
        down_data = new_pred - 1.96*(data[:,-1]-new_pred)
    else:
        new_data = deepcopy(data)
        new_data[0,-1] = up_data; pred_up = Model.predict(new_data)
        new_data[0,-1] = down_data; pred_down = Model.predict(new_data)
        up_data = pred_up + 1.96*(up_data-pred_up)
        down_data = pred_down - 1.96*(down_data-pred_down)
    if(norm):
        pred_60min.append(new_pred[:,2])
        upper_limit.append(up_data)
    else:
        P = scaler.inverse_transform(new_pred)
        pred_60min.append(P[:,2])
        upper_limit.append(scaler.inverse_transform(up_data)[:,2])
        lower_limit.append(scaler.inverse_transform(down_data)[:,2])
    data = np.roll(data,-1,axis=1)
    data[0,-1] = new_pred
""" 
#second method to find confidence interval range
for i in range(pred_len):
    if(i==0):
        data = First_data
    new_pred = Model.predict(data)
    if(norm):
        pred_60min.append(new_pred[:,2])
    else:
        P = scaler.inverse_transform(new_pred)
        pred_60min.append(P[:,2])
    up_data = pred_60min[-1] + 1.96*np.sqrt(np.square(scaler.inverse_transform(data[0,-(i+1):])[:,2]-np.array(pred_60min)[:,0]).sum()) 
    down_data = pred_60min[-1] - 1.96*np.sqrt(np.square(scaler.inverse_transform(data[0,-(i+1):])[:,2]-np.array(pred_60min)[:,0]).sum()) 
    upper_limit.append(up_data)
    lower_limit.append(down_data)
    data = np.roll(data,-1,axis=1)
    data[0,-1] = new_pred
plt.figure()
range_history = First_data.shape[1]
range_future = list(range(range_history, range_history +len(pred_60min)))
if((norm)):
    plt.plot(np.arange(range_history), np.array(First_data[0,:,2]).reshape(-1))
    plt.plot(range_future, np.array(pred_60min).reshape(-1))
    plt.plot(range_future, testing_data[-pred_len-x_test.shape[1]*(index-1):-x_test.shape[1]*(index-1),2].reshape(-1))
else:
    F = scaler.inverse_transform(First_data.reshape(-1,3))
    T = scaler.inverse_transform(testing_data)
    plt.plot(np.arange(range_history), np.array(F[:,2]).reshape(-1))
    plt.plot(range_future, np.array(pred_60min).reshape(-1))
    plt.plot(range_future, T[-pred_len-x_test.shape[1]*(index-1):-x_test.shape[1]*(index-1),2].reshape(-1))    
#plt.plot(range_future, testing_data[x_test.shape[1]:x_test.shape[1]+pred_len,2].reshape(-1))
plt.fill_between(range_future, np.array(lower_limit).reshape(-1), np.array(upper_limit).reshape(-1), color='b', alpha=.1)
plt.legend(['previous data', 'prediction','original'], loc='upper left')
plt.ylabel('(mg/dL)'); plt.xlabel('time/5min')
plt.title(model_name+' 60 minutes blood glucose prediction')
#plt.savefig(filename+'/figures/'+model_name+'_5min_'+'60pred.pdf', format='pdf')