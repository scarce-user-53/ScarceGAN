import sys
from configuration import *
from utils import *
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers
import numpy as np
import pandas as pd
import boto3
import random
import io


s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)

data_unlabeled = read_user_data(s3_location_input_data_healthy)
check_null_values(data_unlabeled)

data_risky= read_user_data(s3_location_input_data_risky)
check_null_values(data_risky)

data_supervised = read_user_data(s3_location_input_labeled_data)
check_null_values(data_supervised)

data_unlabeled = data_unlabeled.fillna(0)
data_risky = data_risky.fillna(0)
data_supervised = data_supervised.fillna(0)

data_risky.columns = data_unlabeled.columns[0:51]
cols_reqd = data_unlabeled.columns[1:51]
# print(cols_reqd)


data_set2 = read_user_data(s3_location_test_set2)
print('data_set2',data_set2.shape)
data_set3 = read_user_data(s3_location_test_set3)
print('data_set3',data_set3.shape)

data_desperate = read_user_data(s3_location_input_desperate_data)
data_desperate = data_desperate.fillna(0)
print('data_desperate',data_desperate.shape)

data_production = read_user_data(s3_location_input_data_production)
data_production = data_production.fillna(0)
print('data_production',data_production.shape)

data_dormant = data_supervised[data_supervised['_2'] == 'Dormant'].iloc[:,3:]
data_dormant.columns = cols_reqd
data_normal = data_supervised[data_supervised['_2'] == 'Normal'].iloc[:,3:]
data_normal.columns = cols_reqd
data_heavy = data_supervised[data_supervised['_2'] == 'Heavy'].iloc[:,3:]
data_heavy.columns = cols_reqd

#print(data_dormant.shape,data_normal.shape,data_heavy.shape)
#(790, 53) (651, 53) (340, 53)

data_dormant = data_dormant[cols_reqd]
data_normal = data_normal[cols_reqd]
data_heavy = data_heavy[cols_reqd]
data_risky = data_risky[cols_reqd]
data_unlabeled = data_unlabeled[cols_reqd]


train_dormant_2 = data_dormant.iloc[:-150,:]
test_dormant = data_dormant.iloc[-150:,:]

train_normal_2 = data_normal.iloc[:-150,:]
test_normal = data_normal.iloc[-150:,:]

train_heavy_2 = data_heavy.iloc[:-100,:]
test_heavy = data_heavy.iloc[-100:,:]

train_risky_2 = data_risky.iloc[:-50,:]
test_risky = data_risky.iloc[-50:,:]




train_dormant_1,test_dormant_1 ,y_dormant , test_y_dormant_1= train_test_split(np.array(train_dormant_2), np.zeros((train_dormant_2.shape[0]), dtype=int), test_size=0.67, random_state=42)
train_normal_1,test_normal_1 ,y_normal, test_y_normal_1 = train_test_split(np.array(train_normal_2), np.ones((train_normal_2.shape[0]), dtype=int), test_size=0.6, random_state=42)
train_heavy_1,test_heavy_1 ,y_heavy , test_y_heavy_1 = train_test_split(np.array(train_heavy_2), np.ones((train_heavy_2.shape[0]), dtype=int)*2, test_size=0.25, random_state=42)
train_risky_1,test_risky_1 ,y_risky , test_y_risky_1 = train_test_split(np.array(train_risky_2), np.ones((train_risky_2.shape[0]), dtype=int)*3, test_size=0.02, random_state=42)

train_dormant = pd.DataFrame(train_dormant_1)
train_normal = pd.DataFrame(train_normal_1)
train_heavy= pd.DataFrame(train_heavy_1)
train_risky= pd.DataFrame(train_risky_1)

# test_dormant = pd.DataFrame(test_dormant_1)
# test_normal = pd.DataFrame(test_normal_1)
# test_heavy= pd.DataFrame(test_heavy_1)
# test_risky= pd.DataFrame(test_risky_1)

data_unlabeled = data_unlabeled[cols_reqd]

print("Labeled_Data_Points",train_dormant.shape[0]+train_normal.shape[0]+train_heavy.shape[0]+train_risky.shape[0])
print("Test_Data_Points",test_dormant.shape[0]+test_normal.shape[0]+test_heavy.shape[0]+test_risky.shape[0])

train_1 = train_dormant.append(train_normal)
train_2 = train_1.append(train_heavy)
train_3 = train_2.append(train_risky)

###CheckPoint -> Use of scaler only on supervised data
train = data_unlabeled[cols_reqd].append(train_3)

test_1 = test_dormant.append(test_normal)
test_2 = test_1.append(test_heavy)
test = test_2.append(test_risky)



scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_3)


joblib.dump(scaler, scaler_filename) 
# unlabeled_data_1 = create_unlabeled_data(np.array(data_unlabeled[cols_reqd]),np.array(read_user_data(s3_location_input_data_production)),0.5)
unlabeled_data_1 = create_unlabeled_data_v3(np.array(data_unlabeled[cols_reqd]),np.array(read_user_data(s3_location_input_data_production)),np.array(read_user_data(s3_location_input_data_highscores)),[0.5,0.3,0.2])



# unlabeled_data_1 = data_unlabeled[cols_reqd]
unlabeled_data_1 = unlabeled_data_1.reset_index(drop=True)
unlabeled_data = scaler.transform(unlabeled_data_1)
print(unlabeled_data.shape)

test_data = test.copy()
# unlabeled_data = unlabeled_data[200:]
x_dormant = scaler.transform(train_dormant)
x_normal = scaler.transform(train_normal)
x_heavy = scaler.transform(train_heavy)
x_risky = scaler.transform(train_risky)

print(len(y_dormant), len(y_normal), len(y_heavy), len(y_risky))


scaled_test_data = scaler.transform(test_data)
test_y_dormant = np.zeros((test_dormant.shape[0]), dtype=int)
test_y_normal = np.ones((test_normal.shape[0]), dtype=int)* 1
test_y_heavy = np.ones((test_heavy.shape[0]), dtype=int) * 2
test_y_risky = np.ones((test_risky.shape[0]), dtype=int) * 3
y_test = np.concatenate([test_y_dormant,test_y_normal,test_y_heavy,test_y_risky],axis=0)

print(scaled_test_data.shape, len(y_test))

# test_selecteddata = pd.read_csv('Network_Tuning_Data/Scaled_test_set_corrected_Raw_1.csv')
# test_selecteddata_scaled = scaler.fit_transform(test_selecteddata)