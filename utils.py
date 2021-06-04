from configuration import *
# from model_V4 import *
# from data_collection import *
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from tensorflow.keras import initializers
import numpy as np
import pandas as pd
import boto3
import random
import io


s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)



def read_user_data(prefix_):
    prefix_objs = bucket.objects.filter(Prefix=prefix_)
    df = pd.DataFrame()
    file_counter = 0
    for obj in prefix_objs:
        x = obj.get()['Body'].read()
        if (not(obj.key.endswith('/') or obj.key.endswith('_SUCCESS') or not x)):
            file_counter = file_counter +1
            key = obj.key
            body = obj.get()['Body'].read()
            df0 = pd.read_csv(io.BytesIO(body))
            df = df.append(df0)
    return df

def check_null_values(data):
    column_select_list = data.iloc[:,0:51].isna().sum() > 0
    columns = data.columns
    null_columns = [x for x, y in zip(columns, column_select_list) if y == True]
    #print(str(data[null_columns].isna().sum()/data.shape[0] * 100))
    

def create_unlabeled_data(data_1,data_2,percentage = 0.5):
    sample_1_idx = random.sample(list(range(0,data_1.shape[0])),int(percentage * data_1.shape[0]))
    sample_1 = pd.DataFrame(data_1[sample_1_idx])
    sample_2_idx = random.sample(list(range(0,data_2.shape[0])),int((1-percentage) * data_2.shape[0]))
    sample_2 = pd.DataFrame(data_2[sample_2_idx])
    data = sample_1.append(sample_2)
    data.fillna(0,inplace=True)
    return data.iloc[:,1:]


def create_unlabeled_data_v3(data_1,data_2,data_3,percentage = [0.5,0.3,0.2]):
    sample_1_idx = random.sample(list(range(0,data_1.shape[0])),int(percentage[0] * data_1.shape[0]))
    sample_1 = pd.DataFrame(data_1[sample_1_idx])
    
    sample_2_idx = random.sample(list(range(0,data_2.shape[0])),int(percentage[1] * data_2.shape[0]))
    sample_2 = pd.DataFrame(data_2[sample_2_idx])
    
    sample_3_idx = random.sample(list(range(0,data_3.shape[0])),int(percentage[2] * data_3.shape[0]))
    sample_3 = pd.DataFrame(data_3[sample_3_idx])
    sample_3 = sample_3.iloc[:,1:-25]
    
    data_1 = sample_1.append(sample_2)
    data = data_1.append(sample_3)
    data.fillna(0,inplace=True)
    
    return data.iloc[:,1:]


def predict_on_labeled_test_set(scaled_test_data,actual_class,best_model):
    predict_class = best_model.discriminator_supervised.predict_classes(scaled_test_data)
    print(metrics.confusion_matrix(actual_class, predict_class, labels=[0,1,2,3,4]))
    print(metrics.classification_report(actual_class, predict_class, labels=[0,1,2,3,4]))
    return best_model.discriminator_supervised.predict_classes(scaled_test_data),best_model.discriminator_supervised.predict(scaled_test_data)
        
def predict_on_unlabeled_test_set(scaled_test_data,best_model):
    check = pd.DataFrame(best_model.discriminator_supervised.predict_classes(scaled_test_data),columns = ["Class"])
    print("Predicted Dormant User:",check[check["Class"] == 0].shape[0])
    print("Predicted Normal User:",check[check["Class"] == 1].shape[0])
    print("Predicted Heavy User:",check[check["Class"] == 2].shape[0])
    print("Predicted Risky User:",check[check["Class"] == 3].shape[0])
    print("Predicted Unknown User:",check[check["Class"] == 4].shape[0])
    
    return best_model.discriminator_supervised.predict(scaled_test_data)

def get_prediction_test_set(location,class_notation,model):
    scaler = joblib.load(scaler_filename)
    data = read_user_data(location)
    data.fillna(0,inplace = True)
    data = scaler.transform(data.iloc[:,1:51]) 
    check = pd.DataFrame(model.predict_classes(data),columns = ["Class"])
    return check[check["Class"] == class_notation].shape[0]


def add_data(data_frame,model,item,data,y):
    
    predict_class = model.predict_classes(data)
    cm = metrics.confusion_matrix(y, predict_class, labels=[0,1,2,3])
    data_frame=data_frame.append(create_row(cm,item,model,'model'), ignore_index=True)
    
    best_model = SemiSupervisedGAN()
    best_model.discriminator_supervised.load_weights(saved_model_location)
    predict_class = best_model.discriminator_supervised.predict_classes(data)
    cm = metrics.confusion_matrix(y, predict_class, labels=[0,1,2,3])
    data_frame=data_frame.append(create_row(cm,item,best_model.discriminator_supervised,'best_model'), ignore_index=True)
    
    return data_frame