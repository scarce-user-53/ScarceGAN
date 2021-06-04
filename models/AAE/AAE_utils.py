from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import boto3
import io
from sklearn.externals import joblib

s3 = boto3.resource('s3')
bucket = s3.Bucket('bucket_name')


def read_user_data(prefix_):
    prefix_objs = bucket.objects.filter(Prefix=prefix_)
    df = pd.DataFrame()
    file_counter = 0
    for obj in prefix_objs:
        x = obj.get()['Body'].read()
        if (not(obj.key.endswith('/') or obj.key.endswith('_temporary') or not x)):
            file_counter = file_counter +1
            key = obj.key
            body = obj.get()['Body'].read()
            df0 = pd.read_csv(io.BytesIO(body))
            df = df.append(df0)
    return df


s3_location_input_data = 'Surajit/MARCO/Processed_TimeSeries_DataFolder/Verify_VAE_Deployed/Features_Prophet_DTW/Assumed_Healthy_Users/'
s3_location_input_data_risky= 'Surajit/MARCO/Processed_TimeSeries_DataFolder/Validation_Prophet_Features_VAE_Process_Date/'
validation_users_healthy = 20000

data_train = read_user_data(s3_location_input_data)
data_risky = read_user_data(s3_location_input_data_risky)
data_risky = data_risky.fillna(0)
data_train = data_train.fillna(0)
cols_reqd = data_train.columns[1:-25]

scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train[cols_reqd])

scaler_filename = "Scaler_AAE.save"
joblib.dump(scaler, scaler_filename) 

training_data = data_train_scaled[0:-validation_users_healthy,:]
valid_data = data_train_scaled[-validation_users_healthy:,:]
risky_positive_data = scaler.transform(data_risky.iloc[:100,1:])

x_train = training_data
x_valid = valid_data
x_risky = risky_positive_data

x_train = np.array(x_train)
x_valid = np.array(x_valid)
x_risky = np.array(x_risky)
