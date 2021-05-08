import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

#file location of dataset
filename = 

df1 = pd.read_csv(filename+"heart_rate_data.csv")
#type(df.point_timestamp[0])
#sum(dataset[:,2]==-700)
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

# merge 3 features based blood suger
merge1 = pd.merge(df3, df2[['point_value(kilometers)','point_timestamp']], on=['point_timestamp'], how='left')
merge2 = pd.merge(merge1, df1[['point_value','point_timestamp']], on=['point_timestamp'], how='left')
print('number of distance nan values: ',merge2['point_value(kilometers)'].isnull().values.sum())
print('number of heart rate nan values: ',merge2['point_value'].isnull().values.sum())
merge2.point_value.interpolate(inplace = True) #interpolate nan values
merge2['point_value(kilometers)'].interpolate(inplace = True)
merge2.dropna(inplace=True) # Drop all rows with missing values
merge2.drop_duplicates(subset=['point_timestamp'],inplace=True) #drop duplicate data
print(merge2.head())

Date = pd.to_datetime(df3['point_timestamp'].str.split(" ",expand=True).iloc[:,0])
Time = (df3['point_timestamp'].str.split(" ",expand=True).iloc[:,1])
train_index = (Date >= '2017-05-15') & (Date < '2017-05-30')
valid_index = (Date >= '2017-05-30') & (Date < '2017-06-14')
test_index = (Date >= '2017-06-14')

df3['Date'] = Date
df3['month'] = Date.dt.month; dataset_by_month = df3.resample('M',on='Date').sum()

plt.figure();plt.plot(df3.index,df3['point_value(mg/dL)'].values)
plt.ylabel('blood glucose(mg/dL)');plt.xlabel('Time(day)')

def create_data(Data,timelap=60):
    X,Y = [], []
    for i in range(Data.shape[0]-timelap):
        X.append(Data[i:i+timelap])
        Y.append(Data[i+timelap])
    return np.array(X),np.array(Y)

lap=24
dataset = df3["point_value(mg/dL)"].values

maxmin=True
if(maxmin):
    Min,Max = np.min(dataset[train_index]),np.max(dataset[train_index])
    dataset = (dataset-Min)/(Max-Min)
else:
    Mean,std = np.mean(dataset[train_index]),np.std(dataset[train_index])
    dataset = (dataset-Mean)/(std)

x_train, y_train = create_data((dataset[train_index]),timelap=lap)
x_test, y_test = create_data((dataset[test_index]),timelap=lap)
x_valid, y_valid = create_data((dataset[valid_index]),timelap=lap)

def lstm(units,X):
    model = Sequential()
    model.add(LSTM(units = units, return_sequences=True, 
              input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units = units))
    model.add(Dense(1))
    #Compile model
    optimizer = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=optimizer,loss='mse')
    return model

def gru(units,X):
    model = Sequential()
    model.add(GRU(units = units, return_sequences = True, 
                   input_shape = [X.shape[1], X.shape[2]]))
    #model.add(Dropout(0.5)) 
    model.add(GRU(units = units)) 
    #model.add(Dropout(0.5))
    model.add(Dense(units = 1)) 
    #Compiler
    optimizer = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=optimizer,loss='mse')
    return model

if(False):
    Model = gru(1,x_train.reshape(-1,x_train.shape[1],1))
    model_name = 'GRU_model'
else:
    Model = lstm(1,x_train.reshape(-1,x_train.shape[1],1))
    model_name = 'LSTM_model'

early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                           patience = 15)
history = Model.fit(x_train.reshape(-1,x_train.shape[1],1), y_train, epochs = 200,  
                    validation_data=(x_valid.reshape(-1,x_train.shape[1],1), y_valid),
                    batch_size = 32, shuffle = False, 
                    callbacks = [early_stop])
plt.figure();
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Train vs Validation Loss for bilstm')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['Train loss', 'Validation loss'], loc='upper right')
#plt.savefig(filename+'/figures/'+model_name+'_5min_val_train_loss_1feature.pdf', format='pdf')

# denormalizing the data
def denorm(x,norm=False):
    if(maxmin):
        if(norm):
            x = (x - np.min(x))/(np.max(x)-np.min(x))
        return x*(Max-Min) + Min
    else:
        return x*std+Mean

pred = Model.predict(x_test.reshape(-1,x_train.shape[1],1))
Error = denorm(pred,True) - denorm(y_test)
mse = np.square(Error).mean()
rmse = np.sqrt(mse)
mae = np.abs(Error).mean()
print(model_name+' GRU Mean Absolute Error: {:.3f}'.format(mae))
print(model_name+' GRU Root Mean Square Error: {:.3f}'.format(rmse))

plt.figure()
plt.plot(denorm(pred,True))
plt.plot(denorm(y_test))
plt.legend(['prediction','original'],loc='upper left')
#plt.savefig(filename+'/figures/'+model_name+'_5min_'+'5pred_1feature.pdf', format='pdf')

plt.figure()
plt.plot(denorm(pred,True)[:36])
plt.plot(denorm(y_test)[:36])
plt.legend(['prediction','original'],loc='upper left')
#plt.savefig(filename+'/figures/'+model_name+'_5min_first_three_hours_'+'5pred_1feature.pdf', format='pdf')

# 60 min prediction plot
norm = False
pred_60min = [] ; upper_limit = []; lower_limit = []
pred_len = 60//5
testing_data = (dataset[test_index])
index = 2
First_data = testing_data[-pred_len-x_test.shape[1]*(index):-pred_len-x_test.shape[1]*(index-1)].reshape(1,-1,1)
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
        pred_60min.append(new_pred)
        upper_limit.append(up_data)
        lower_limit.append(down_data)
    else:
        P = denorm(new_pred)
        pred_60min.append(P)
        upper_limit.append(denorm(up_data))
        lower_limit.append(denorm(down_data))
    data = np.roll(data,-1,axis=1)
    data[0,-1] = new_pred
""" 
#second method to find confidence interval range
for i in range(pred_len):
    if(i==0):
        data = First_data
    new_pred = Model.predict(data)
    if(norm):
        pred_60min.append(new_pred)
    else:
        P = denorm(new_pred)
        pred_60min.append(P)
    up_data = pred_60min[-1] + 1.96*np.sqrt(np.square(denorm(data[0,-(i+1):])-np.array(pred_60min)).sum()) 
    down_data = pred_60min[-1] - 1.96*np.sqrt(np.square(denorm(data[0,-(i+1):])-np.array(pred_60min)).sum()) 
    upper_limit.append(up_data)
    lower_limit.append(down_data)
    data = np.roll(data,-1,axis=1)
    data[0,-1] = new_pred
#"""
plt.figure()
range_history = First_data.shape[1]
range_future = list(range(range_history, range_history +len(pred_60min)))
if((norm)):
    plt.plot(np.arange(range_history), np.array(First_data[0]).reshape(-1))
    plt.plot(range_future, np.array(pred_60min).reshape(-1))
    plt.plot(range_future, testing_data[-pred_len-x_test.shape[1]*(index-1):-x_test.shape[1]*(index-1)].reshape(-1))
else:
    F = denorm(First_data.reshape(-1,3))
    T = denorm(testing_data)
    plt.plot(np.arange(range_history), np.array(F[:]).reshape(-1))
    plt.plot(range_future, np.array(pred_60min).reshape(-1))
    plt.plot(range_future, T[-pred_len-x_test.shape[1]*(index-1):-x_test.shape[1]*(index-1)].reshape(-1))    
plt.fill_between(range_future, np.array(lower_limit).reshape(-1), np.array(upper_limit).reshape(-1), color='b', alpha=.1)
plt.legend(['previous data', 'prediction','original'], loc='upper left')
plt.ylabel('(mg/dL)'); plt.xlabel('time/5min')
plt.title(model_name+' 60 minutes blood glucose prediction')
#plt.savefig(filename+'/figures/'+model_name+'_5min_'+'60pred_1feature.pdf', format='pdf')
