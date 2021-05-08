import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, metrics

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

# round the min to find more time match between 3 data
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
pd.set_option('display.max_columns', None)
print(merge2.head())

Date = pd.to_datetime(merge2['point_timestamp'].str.split(" ",expand=True).iloc[:,0])
train_index = (Date >= '2017-05-16') & (Date < '2017-06-01')
valid_index = (Date >= '2017-06-01') & (Date < '2017-06-15')
test_index = (Date >= '2017-06-15')

merge2['Date'] = Date
merge2['month'] = Date.dt.month; dataset_by_month = merge2.resample('M',on='Date').sum()

weekrange = (np.arange(merge2.shape[0])+1)*5/60/7
plt.figure();plt.plot(weekrange,merge2['point_value(mg/dL)'].values)
plt.ylabel('blood glucose(mg/dL)');plt.xlabel('Time/week');plt.title('weekly observation')
plt.savefig(filename+'/figures/'+'weekly_data.pdf', format='pdf')

oneday = (np.arange(24*12)+1)*5/60; ind = 2 #one day plot
plt.figure();plt.plot(oneday,merge2['point_value(mg/dL)'].values[(ind-1)*len(oneday):ind*len(oneday)])
plt.ylabel('blood glucose(mg/dL)');plt.xlabel('hours');plt.title('24 hour of data')
plt.savefig(filename+'/figures/'+'one_day_data.pdf', format='pdf')

plt.figure()
plt.hist2d(merge2['point_value'].values, merge2['point_value(mg/dL)'].values, bins=(50, 50))
plt.colorbar();plt.xlabel('kilometers');plt.ylabel('glucose level')

def create_data(Data,timelap=24,y_len=12):
    X,Y = [], []
    for i in range(Data.shape[0]-timelap-y_len):
        X.append(Data[i:i+timelap])
        Y.append(Data[i+timelap:i+timelap+y_len])
    return np.array(X),np.array(Y)

lap=24
dataset = merge2[["point_value(kilometers)","point_value","point_value(mg/dL)"]].values

"""
import seaborn as sns
ax = sns.boxplot(x="feature", y="", data=merge2[["point_value(kilometers)","point_value","point_value(mg/dL)"]])
#"""

x_train, y_train = create_data((dataset[train_index]),timelap=lap)
x_test, y_test = create_data((dataset[test_index]),timelap=lap)
x_valid, y_valid = create_data((dataset[valid_index]),timelap=lap)

#normalize features with max-min
def Norm(Min,Max,X,denorm,normalize=False):
    if(denorm):
        return X*(Max-Min) + Min
    else:
        return (X-Min)/(Max-Min)
maxmin = True
if(maxmin):
    scaler = preprocessing.MinMaxScaler().fit(x_train.reshape(-1,3))
    Min,Max = np.min(x_train.reshape(-1,3),axis=0)[2],np.max(x_train.reshape(-1,3),axis=0)[2]
    x_train = scaler.transform(x_train.reshape(-1,3)).reshape(len(y_train),-1,3)
    x_test = scaler.transform(x_test.reshape(-1,3)).reshape(len(y_test),-1,3)
    x_valid = scaler.transform(x_valid.reshape(-1,3)).reshape(len(y_valid),-1,3)
    y_train,y_test,y_valid = Norm(Min,Max,y_train,False),Norm(Min,Max,y_test,False),Norm(Min,Max,y_valid,False)
   
#normalize time-series data by mean std
meanstd = False
if(meanstd):
    scaler2 = preprocessing.StandardScaler().fit(x_train[:,:,2])
    x_train[:,:,2] = scaler2.transform(x_train[:,:,2])
    x_test[:,:,2] = scaler2.transform(x_test[:,:,2])
    x_valid[:,:,2] = scaler2.transform(x_valid[:,:,2])
    

def lstm_60(X,units=12):
    model = Sequential()
    model.add(LSTM(units = units, return_sequences=True,
                   input_shape=(X.shape[1], X.shape[2]))) 
    model.add((LSTM(units = units)))
    model.add(Dense(units))
    #Compiler
    optimizer = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=optimizer,loss='mse')
    return model

def gru_60(X,units=12):
    model = Sequential()
    model.add(GRU(units = units,return_sequences = True, 
                   input_shape = [X.shape[1], X.shape[2]]))
    #model.add(Dropout(0.5)) 
    model.add(GRU(units = units,return_sequences = True))
    #model.add(Dropout(0.5)) 
    model.add(GRU(units = units))  
    model.add(Dense(units))
    #Compiler
    optimizer = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=optimizer,loss='mse')
    return model

if(True):
    Model = gru_60(x_train)
    model_name = 'GRU_model'
else:
    Model = lstm_60(x_train)
    model_name = 'LSTM_model'

early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                           patience = 15)
history = Model.fit(x_train, y_train[:,:,2], epochs = 200,  
                    validation_data=(x_valid, y_valid[:,:,2]),
                    batch_size = 32, shuffle = False, 
                    callbacks = [early_stop])
plt.figure();
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(model_name+' train vs validation loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['Train loss', 'Validation loss'], loc='upper right')
#plt.savefig(filename+'/figures/'+model_name+'_60min_val_train_loss.pdf', format='pdf')
print(Model.summary())

pred_norm = Model.predict(x_test)
if(maxmin):
    pred,y_test_denorm = Norm(Min,Max,pred_norm,True,True),Norm(Min,Max,y_test[:,:,2],True)
Error = pred - y_test_denorm
mse = np.square(Error).mean()
#mse = metrics.mean_squared_error(y_test_denorm,pred,squared=True)
rmse = np.sqrt(mse)
mae = np.abs(Error).mean()
print(model_name+' Mean Absolute Error: {:.3f}'.format(mae))
print(model_name+' Root Mean Square Error: {:.3f}'.format(rmse))

norm = False
plt.figure()
index = -1*24
range_history = x_test.shape[1]
range_future = list(range(range_history, range_history +len(pred[index])))
if((norm)):
    plt.plot(np.arange(range_history), np.array(x_test[index,:,2]).reshape(-1))
    plt.plot(range_future, np.array(pred_norm[index]).reshape(-1))
    plt.plot(range_future, y_test[index,:,2].reshape(-1))   
else:
    F = scaler.inverse_transform(x_test[index])
    if(meanstd):
        F = scaler2.inverse_transform(F[:,2].reshape(1,-1))
        plt.plot(np.arange(range_history), np.array(F).reshape(-1))
    else:
        plt.plot(np.arange(range_history), np.array(F[:,2]).reshape(-1))
    plt.plot(range_future, np.array(pred[index]).reshape(-1))
    plt.plot(range_future, y_test_denorm[index].reshape(-1))    
#plt.plot(range_future, testing_data[x_test.shape[1]:x_test.shape[1]+pred_len,2].reshape(-1))
u = pd.DataFrame(np.concatenate([F[-1,2].reshape(1),pred[index]]))
path_deviation = 1.96 * u.rolling(2).std()
under_line = (pred[index]-path_deviation.values[1:,0])
over_line = (pred[index]+path_deviation.values[1:,0])
plt.fill_between(range_future, under_line, over_line, color='b', alpha=.1)
plt.legend(['previous data', 'prediction','original'], loc='upper left')
plt.ylabel('(mg/dL)'); plt.xlabel('time/5min')
plt.title(model_name+'60 minutes blood glucose prediction')
#plt.savefig(filename+'/figures/'+model_name+'_60min_'+'60pred.pdf', format='pdf')