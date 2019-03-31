#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import csv
from sklearn import preprocessing
from keras import losses,optimizers
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#讀檔
def readTrain():
    train = pd.read_csv('ElectricPower.csv',usecols=["日期", "尖峰負載(MW)", "備轉容量(MW)", "備轉容量率(%)"])
    train1 = pd.read_csv('ElectricPowerMar.csv',usecols=["日期", "尖峰負載(MW)", "備轉容量(MW)", "備轉容量率(%)"])    
    train = pd.concat([train,train1], axis=0)
    train = pd.DataFrame(train.values,columns=["日期", "尖峰負載(MW)", "備轉容量(MW)", "備轉容量率(%)"])    
    train.fillna(value = 0,inplace=True)
    return train


# In[3]:


#日期格式化
def augFeatures(train):
    train["日期"] = pd.to_datetime(train["日期"], format = '%Y%m%d')
    train["year"] = train["日期"].dt.year
    train["month"] = train["日期"].dt.month
    train["date"] = train["日期"].dt.day
    train["day"] = train["日期"].dt.dayofweek
    train = train.drop(["日期"], axis=1)
    return train


# In[4]:


#資料正規化
def normalize(train_Aug):
    newdf= train_Aug.copy()
    min_max_scaler = preprocessing.MinMaxScaler()
    for col in train_Aug.columns:
         newdf[col] = min_max_scaler.fit_transform(train_Aug[col].values.reshape(-1,1))        
    return newdf

#資料正規化還原
def denormalize(train_Aug, norm_value):
    original_value = train_Aug['尖峰負載(MW)'].values.reshape(-1,1)
    norm_value = norm_value.reshape(-1,1)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm_value = min_max_scaler.inverse_transform(norm_value)        
    return denorm_value


# In[5]:


#用前7天預測後面7天，build training data
def buildTrain(train, pastDay=7, futureDay=7):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay+1):
        X_train.append(np.array(train.iloc[i:i+pastDay]))
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["尖峰負載(MW)"]))
    return np.array(X_train), np.array(Y_train)


# In[6]:


#資料亂序
def shuffle(X,Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]


# In[7]:


#把資料分成training和validation
def splitData(X,Y,rate):
    X_train = X[0:int(X.shape[0]*(1-rate))]
    Y_train = Y[0:int(Y.shape[0]*(1-rate))]
    X_val = X[int(X.shape[0]*(1-rate)):]
    Y_val = Y[int(Y.shape[0]*(1-rate)):]
    return X_train, Y_train, X_val, Y_val


# In[8]:


#LSTM model
def buildManyToManyModel(shape):
    model = Sequential()
    model.add(LSTM(10, input_shape=(shape[1],shape[2]), return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss="mse", optimizer="Adam")
    model.summary()
    return model


# In[9]:


train = readTrain()
train_Aug = augFeatures(train)
train_norm = normalize(train_Aug)
X_train, Y_train = buildTrain(train_norm, 7, 7 )
#X_train, Y_train = shuffle(X_train, Y_train)
X_train, Y_train, X_val, Y_val = splitData(X_train,Y_train,0.1)

#把Y從2維變3維
Y_train = Y_train[:,:,np.newaxis]
Y_val = Y_val[:,:,np.newaxis]

#資料放入model做訓練
model = buildManyToManyModel(X_train.shape)
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])


# In[10]:


# #以下註解為train和val的比較圖
# trainPredict = model.predict(X_train)
# testPredict = model.predict(X_val)

# trainPredict = denormalize(train_Aug, trainPredict).reshape(398, 7)[:,0]
# Y_train = denormalize(train_Aug, Y_train).reshape(398, 7)[:,0]
# testPredict = denormalize(train_Aug, testPredict).reshape(45, 7)[:,0]
# Y_val = denormalize(train_Aug, Y_val).reshape(45, 7)[:,0]


# In[11]:


# plt.plot(trainPredict[:50],color='red', label='Prediction')
# plt.plot(Y_train[:50],color='blue', label='Answer')
# plt.legend(loc='best')
# plt.show()


# In[12]:


# plt.plot(testPredict[:50],color='red', label='Prediction')
# plt.plot(Y_val[:50],color='blue', label='Answer')
# plt.legend(loc='best')
# plt.show()


# In[13]:


# val = denormalize(train_Aug, X_val)
# val = np.reshape(X_val[-1], (1,7,7))
# X_val.shape
# res = model.predict(val)
# plt.plot(res.flatten(),color='red', label='Prediction')
# plt.legend(loc='best')
# plt.show()


# In[14]:


#拿出最後7天準備預測未來7天
predictdata = train[-7:].drop(["日期"], axis=1)


# In[15]:


#資料正規化，轉維度，丟入model，正規化還原得到預測值
norm_predictdata = normalize(predictdata)
pre_predictdata = norm_predictdata.values.reshape((1,predictdata.shape[0],predictdata.shape[1]))
res = model.predict(pre_predictdata)
predict = denormalize(predictdata, res)


# In[16]:


#將所得的結果與前2年春假的data取平均
predict[0] = (predict[0]+28161+29047)/3
predict[1] = (predict[1]+28739+29267)/3
predict[2] = (predict[2]+24245+24981)/3
predict[3] = (predict[3]+22905+24450)/3
predict[4] = (predict[4]+22797+23940)/3
predict[5] = (predict[5]+23638+23895)/3
predict[6] = (predict[6]+27722+28232)/3
print (predict)


# In[17]:


#把得到的預測值存入D槽的資料夾jupyter底下
submission = pd.DataFrame(predict, columns=["peak_load(MW)"], index=["20190402","20190403","20190404","20190405","20190406","20190407","20190408"])
submission.to_csv('D:\\jupyter\\submission.csv')


# In[ ]:





# In[ ]:




