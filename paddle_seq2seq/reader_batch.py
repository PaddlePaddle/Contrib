import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import os

X_seq = 20
Y_seq = 3
feature_num = 74

def get_inout(input, output, city, samples, x_seq, y_seq):
    X=list()
    y=list()
    to_x = list(input[:,city])
    to_y = list(output[:,city])
      
    for j in range(samples):
      y.append(to_y[j:j+y_seq])
      X.append(to_x[j:j+x_seq]) 
    return X, y 

def generate_data(input, output, x_seq, y_seq, size):
    X=list()
    y=list()
    for i in range(1,feature_num+1):
      X_app, Y_app = get_inout(input, output, i, size, x_seq, y_seq)
      X.append(X_app)
      y.append(Y_app)
    X, y = np.array(X), np.array(y)
    return X, y

def construct_sample(args):
    city = pd.read_csv("./data/crawl_list.csv")
    city_list = list(city['city'])
    input = pd.read_csv("./data/input.csv",index_col=0)
    input = input.replace(np.nan,0)
    output = pd.read_csv("./data/output.csv",index_col=0)
    data_size = output.shape[0] + 1 - Y_seq
    input = input[-(output.shape[0]-1+X_seq):]
    
    # normalization the data 
    scaler = MinMaxScaler(feature_range=(0, 1))
    input = scaler.fit_transform(input)
    output = scaler.fit_transform(output)

    X,Y = generate_data(input, output, X_seq, Y_seq, data_size)
    X = (X.swapaxes(0,1)).swapaxes(1,2)
    Y = np.reshape(Y, (feature_num, -1, Y_seq))
    Y = (Y.swapaxes(0,1)).swapaxes(1,2)
    return X, Y, data_size 

def train_reader(X, Y, data_size, batch_size):
    train_size = int(data_size)
    X_train, Y_train = X[:train_size,:,:], Y[:train_size,:,:]
    batch_size = batch_size if X_train.shape[0] > batch_size else X_train.shape[0] 
    def __reader__():
        sample_num = batch_size
        count = 0
        while sample_num < X_train.shape[0]:
            data1, data2, data3 = X[count*batch_size:(count+1)*batch_size,:,:],\
                  Y[count*batch_size:(count+1)*batch_size,:,:],\
                  np.arange(0, batch_size+1)
            data1 = data1.reshape((-1, feature_num))
            yield data1.astype('float32'), data2.astype('float32'), (data3*X_seq).astype('int32')
            count += 1 
            sample_num += batch_size
        if batch_size * count < X_train.shape[0]: 
            data1, data2, data3 = X[count*batch_size:,:,:],\
                  Y[count*batch_size:,:,:],\
                  np.arange(0, X_train.shape[0]-count*batch_size+1)
            data1 = data1.reshape((-1, feature_num))
            yield data1.astype('float32'), data2.astype('float32'), (data3*X_seq).astype('int32')
    return __reader__

def val_reader(X, Y, data_size, batch_size):
    train_size = int(data_size)
    X_val, Y_val = X[train_size:,:,:], Y[train_size:,:,:]
    batch_size = batch_size if X_val.shape[0] > batch_size else X_val.shape[0] 
    def __reader__():
        sample_num = batch_size
        count = 0
        while sample_num < X_val.shape[0]:
            data1, data2, data3 = X[count*batch_size:(count+1)*batch_size,:,:],\
                  Y[count*batch_size:(count+1)*batch_size,:,:],\
                  np.arange(0, batch_size+1)
            data1 = data1.reshape((-1, feature_num))
            yield data1.astype('float32'), data2.astype('float32'), (data3*X_seq).astype('int32')
            count += 1 
            sample_num += batch_size
        if batch_size * count < X_val.shape[0]: 
            data1, data2, data3 = X[count*batch_size:,:,:],\
                  Y[count*batch_size:,:,:],\
                  np.arange(0, X_val.shape[0]-count*batch_size+1)
            data1 = data1.reshape((-1, feature_num))
            yield data1.astype('float32'), data2.astype('float32'), (data3*X_seq).astype('int32')
    return __reader__

       
def val_reader(X, Y, data_size, batch_size):
    train_size = int(data_size)
    X_val, Y_val = X[train_size:,:,:], Y[train_size:,:,:]
    batch_size = batch_size if X_val.shape[0] > batch_size else X_val.shape[0] 
    def __reader__():
        sample_num = batch_size
        count = 0
        while sample_num < X_val.shape[0]:
            data1, data2, data3 = X[count*batch_size:(count+1)*batch_size,:,:],\
                  Y[count*batch_size:(count+1)*batch_size,:,:],\
                  np.arange(0, batch_size+1)
            data1 = data1.reshape((-1, feature_num))
            yield data1.astype('float32'), data2.astype('float32'), (data3*X_seq).astype('int32')
            count += 1 
            sample_num += batch_size
        if batch_size * count < X_val.shape[0]: 
            data1, data2, data3 = X[count*batch_size:,:,:],\
                  Y[count*batch_size:,:,:],\
                  np.arange(0, X_val.shape[0]-count*batch_size+1)
            data1 = data1.reshape((-1, feature_num))
            yield data1.astype('float32'), data2.astype('float32'), (data3*X_seq).astype('int32')
    return __reader__

def test_reader():
    csv = pd.read_csv("./data/input.csv",index_col=0)
    input = csv.copy()
    input = input.replace(np.nan,0)
    output = pd.read_csv("./data/output.csv",index_col=0)
    data_size = output.shape[0] + 1 - Y_seq
#     input = input[-(output.shape[0]-1+X_seq):]
    
    # normalization the data 
    scaler = MinMaxScaler(feature_range=(0, 1))
    input = scaler.fit_transform(input)
    inputs = []
    index = []
    for i in range(20-input.shape[0],0):
        inputs.append(input[i-20:i,1:].copy().astype('float32'))
        index.append(csv.index[i-1])
    inputs.append(input[-20:,1:].copy().astype('float32'))  
    index.append(csv.index[-1])
    lod = np.arange(0, 2)
    return inputs, index, (lod*X_seq).astype('int32') 
