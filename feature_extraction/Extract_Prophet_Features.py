#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import boto3
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.serialize import model_to_json, model_from_json


import io
import re
import json
import itertools
import time
# import tqdm
import boto3
import sys
import datetime
from datetime import datetime,timedelta


from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, cpu_count
from scipy import signal


# In[ ]:


from io import StringIO
from csv import writer


# In[ ]:


from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import col
from pyspark.sql import Row


# In[ ]:


import numpy as np
import pandas as pd
from fbprophet import Prophet
from pyspark import SparkConf
from pyspark.sql.functions import collect_list, struct
from pyspark.sql.types import FloatType, StructField, StructType, StringType, TimestampType, DoubleType
from sklearn.metrics import mean_squared_error



# In[ ]:
#pip3 install future
from past.builtins import xrange
import statistics



# # Fixed Model Parameters




changepoint_prior_scale = 0.001
seasonality_prior_scale = 0.01
changepoint_range = 0.5
interval_width = 0.95
weekly_seasonality = False
seasonality_mode = 'additive'
test_days = 5
forecast_days = 10


# # Script Configuration

# In[ ]:

###########################  Select User Population ##################
## Test Set2 User Population

s3_data_loc = "s3a://bucket_name/Output_Raw_Data/"
target_dir ='s3a://bucket_name/GAN_Data/Set2/Prophet_Features/'

########################## Completed Selection of User Population ##################

feature = ['counter_1','cpcounter_2','counter_3','counter_4','counter_5']
date_time_log = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Spark Context Creation and  Spark session initialization
spark = SparkSession.builder.appName("Prophet_Model") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.thriftServer.incrementalCollect", "true") \
    .getOrCreate()

print("##Spark configuration:")
for i in spark.sparkContext.getConf().getAll():
    print(i)

sc = spark.sparkContext

########################### Spark Context Creation Completed ##################


# # Helper Functions 

#DTW features did not boost the metrics hence did not use
def _dtw_distance(ts_a, ts_b, d = lambda x,y: abs(x-y),max_warping_window=1000):
        """Returns the DTW similarity distance between two 2-D
        timeseries numpy arrays.

        Arguments
        ---------
        ts_a, ts_b : array of shape [n_samples, n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared
        
        d : DistanceMetric object (default = abs(x-y))
            the distance measure used for A_i - B_j in the
            DTW dynamic programming function
        
        Returns
        -------
        DTW distance between A and B
        """
        try:
            exception_flag = False
            # Create cost matrix via broadcasting with large int
            ts_a, ts_b = np.array(ts_a), np.array(ts_b)
            M, N = len(ts_a), len(ts_b)
            cost = sys.maxsize * np.ones((M, N))

            # Initialize the first row and column
            cost[0, 0] = d(ts_a[0], ts_b[0])
            for i in xrange(1, M):
                cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

            for j in xrange(1, N):
                cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

            # Populate rest of cost matrix within window
            for i in xrange(1, M):
                for j in xrange(max(1, i - max_warping_window),
                                min(N, i + max_warping_window)):
                    choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
                    cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])
       
        except Exception as e:
            exception_flag = True
            error_detail = 'error while calculating _dtw_distance for user : '
            s3_log_error('_dtw_distance',error_detail+str(e))

        return cost[-1, -1]

def dtw_self_similarity_features(ts_list, partitions):
    try: 
        exception_flag = False
        ind = np.int32(np.floor(len(ts_list) / partitions))

        
        if ind == 0:
            return -1
        breaks = [i*ind  for i in range(1,partitions)]
        start = 0
        series = []

        for markers in breaks:
            series.append(ts_list[start:markers])
            start = markers


        distance = [_dtw_distance(series[i], series[i+1]) for i in range(0,len(series)-1)]
            
        returnable = [ str(np.mean(distance)), str(np.std(distance)),str(statistics.median(distance))
                      ,str(statistics.variance(distance)),str((statistics.stdev(distance)/ statistics.mean(distance))) if statistics.mean(distance) != 0 else None]

     
    except Exception as e:
        exception_flag = True
        error_detail = 'error while calculating self similarity features for user : '
        s3_log_error('dtw_self_similarity_features',error_detail+str(e))
          
    if exception_flag == True:
        return [None,None,None,None,None]
    else:
        return(returnable)
    
def s3_log_error(function_name,error_messege):
    s3 = boto3.resource('s3')
    client = boto3.client('s3')
    bucket = s3.Bucket('bucket_name')
    file_name = 'Error_Log.csv'
    prefix_ = '/Error_Log_Data/'+file_name
    prefix_objs = bucket.objects.filter(Prefix=prefix_)

    for obj in prefix_objs:
        x = obj.get()['Body'].read()
        if (not(obj.key.endswith('/') or obj.key.endswith('_SUCCESS') or not x)):
            body = obj.get()['Body'].read()
            df0 = pd.read_csv(io.BytesIO(body))
            df0 = df0.append({'Message':str(error_messege),'Error_Function_Name':str(function_name),
                              'Date_Time':str(pd.to_datetime(datetime.now())),'Log_Type':'Error'},ignore_index =True)

            csv_buffer = StringIO()
            df0.to_csv(csv_buffer, sep=",", index=False)
            s3.Object('bucket_name', file_name).put(Body=csv_buffer.getvalue(),Key = obj.key)
            
            
def log_error(lambda_function,detail):
    
    error = [(lambda_function,str(pd.to_datetime(datetime.now())), detail)]    #str(error_obj.__class__),
    Columns = ["Function_Name","Date_Time","Detail"]
    errorDF = spark.createDataFrame(data=error, schema = Columns)
    errorDF.write.options(header=True).csv(error_log_dir, mode='append')
#     sc.stop()

def retrieve_data():
    try:
        """Load  data from S3 location as a pyspark.sql.DataFrame."""
        df = (spark.read
              .option("header", "true")
              .option("inferSchema", value=True)
              .csv(s3_data_loc)) 

        
    except Exception as e:
        error_detail = 'error while fetching data'
        s3_log_error('retrieve_data',str(e))
        
        
        
    return df

def is_weekend(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 6 or date.weekday() == 5:
        return 1
    else:
        return 0

def transform_data(row):
    try:
        """Transform data from pyspark.sql.Row to python dict to be used in rdd."""
        exception_flag = False
        data = row['data']
        user_id = row['user_id']


        # Transform [pyspark.sql.Dataframe.Row] -> [dict]
        data_dicts = []
        for d in data:
            data_dicts.append(d.asDict())

        # Convert into pandas dataframe for fbprophet
        data = pd.DataFrame(data_dicts)
        data['time_stamp'] = pd.to_datetime(data['time_stamp']).dt.date

        data = data.sort_values(by="time_stamp",ascending=True)

        #Add Regressor - Time_Difference    
        data['time_stamp_diff'] = pd.to_datetime(data['time_stamp']).diff() / np.timedelta64(1, 'D')
        data['time_stamp_diff'] = data['time_stamp_diff'].replace(np.nan,0)

        #Add Regressor - Weekend
        data['is_weekend'] = data['time_stamp'].apply(is_weekend)

    except Exception as e:
        exception_flag = True
        error_detail =  str(user_id) + " "
        s3_log_error('transform_data',error_detail + str(e))
    
    
    return {
        'user_id': user_id,
        'data': data,
        'exception_raised': exception_flag
    }


def create_upper_matrix(values, size):
    upper = np.zeros((size, size))
    upper[np.triu_indices(size, 0)] = values
    return upper

def partition_data(d):
    
    try: 
        """Split data into training and testing based on timestamp."""
        d['exception_raised'] = False
        meta_data = d['data'].copy()  
        for feature_name in feature:
            data_name = feature_name + '_data'
            column_list = ['time_stamp','is_weekend','time_stamp_diff']
            column_list.append(feature_name)
            d[data_name] = meta_data[column_list].copy()
            d[data_name].rename(columns = {feature_name: 'y','time_stamp': 'ds'}, inplace = True)
            d[data_name] = d[data_name].fillna(0)

    except Exception as e:
        d['exception_raised'] = True
        d['error']= str(e)
        error_detail = str(d['user_id']) + " "#'error while partitioning data for user : '+ 
        s3_log_error('partition_data',error_detail+str(e))
        
        
    return d
    


def create_model(d):
    try:
        """Create Prophet model using each input grid parameter set."""
        d['exception_raised'] = False
        d['params'] = [d['user_id']]

        for feature_name in feature:
            m = Prophet(seasonality_prior_scale=seasonality_prior_scale,
                        changepoint_prior_scale=changepoint_prior_scale,
                        changepoint_range=changepoint_range,
                        interval_width=interval_width, 
                        weekly_seasonality=weekly_seasonality, 
                        daily_seasonality=False,
                        seasonality_mode = seasonality_mode)

            data_name = feature_name + '_data'

            #Adding custom Seasonality
            f, Pxx_den = signal.periodogram(d[data_name].y)
            for index in Pxx_den.argsort()[-3:][::-1]:
                if f[index] > 0:
                    harmonic_period = np.float64(1/f[index])
                    name_ = 'custom_'+str(np.int32(1/f[index]))
                    m.add_seasonality(name=name_, period=harmonic_period, fourier_order=5)
                    d['params'].append(harmonic_period.item())
                else:
                    d['params'].append(0.0)

            #Adding Regressors
            m.add_regressor('is_weekend',prior_scale=0.5, mode='additive')
            m.add_regressor('time_stamp_diff',prior_scale=0.5, mode='multiplicative')

            model_name = 'model_'+ feature_name

            d[model_name] = m

    except Exception as e:
        d['exception_raised'] = True
        error_detail =  str(d['user_id'])+" "#'error while creating model for user : '+
        s3_log_error('create_model',error_detail+str(e))
        
    return d


def train_model(d):
    try:
        """Fit the model using the training data."""
        d['exception_raised'] = False
        for feature_name in feature:
            model_name = 'model_'+ feature_name
            data_name = feature_name + '_data'
            model = d[model_name]
            train_data = d[data_name]
            model.fit(train_data)
            d[model_name] = model
    except Exception as e:
        d['exception_raised'] = True
        error_detail =  str(d['user_id'])+ " " #'error while training model for user : '+
        s3_log_error('train_model',error_detail+str(e))

    return d

def extract_model_parameters(d):
    
    try:
        d['exception_raised'] = False
        params =[]

        for feature_name in feature:
            model_name = 'model_'+ feature_name
            data_name = feature_name + '_data'
            parameter_name = 'BaseRate_'+ feature_name


            model = d[model_name]
            model_json = json.loads(model_to_json(model))
            train_data = d[data_name]

            #Average offset
            offset = np.mean(np.array(model_json['params']['m']))
            params.append(np.nan_to_num(offset).item())

            #Average Baserate
            base_rate = np.mean(np.array(model_json['params']['k']))
            params.append(np.nan_to_num(base_rate).item())

            #Average days between changepoints
            cp_index = np.array(json.loads(model_json['changepoints'])['index'])
            cp_index_u = cp_index[1:]
            cp_index_l = cp_index[0:-1]
            avg_cp_rate = np.mean(cp_index_u - cp_index_l)
            params.append(np.nan_to_num(avg_cp_rate).item())
            std_cp_rate = np.std(cp_index_u - cp_index_l)
            params.append(np.nan_to_num(std_cp_rate).item())

            ##Growth_Rate
            growth_cp = pd.DataFrame()

            Avg_Growth_rate =[]
            Std_Growth_rate =[]
            Avg_delta = []

            for delta in model_json['params']['delta']:
                n = len(delta)
                matrix = create_upper_matrix(np.ones(int((n**2-n)/2+n)), n)
                x = np.array(delta).reshape(1,len(delta))
                #print(np.dot(x,matrix))
                Avg_Growth_rate.append(np.mean(np.dot(x,matrix)))
                Std_Growth_rate.append(np.std(np.dot(x,matrix)))
                Avg_delta.append(np.mean(np.array(delta)))

            growth_cp['Average']= Avg_Growth_rate
            growth_cp['Std_Dev']= Std_Growth_rate
            growth_cp['Avg_Delta'] = Avg_delta 

            params.append(np.nan_to_num(growth_cp.mean()[0]).item())# Avg Growth rate
            params.append(np.nan_to_num(growth_cp.mean()[1]).item())# Std Growth rate
            params.append(np.nan_to_num(growth_cp.mean()[2]).item())# Laplace Parameter
            
            ## Adding DTW Features in Parameters
            #returnable = dtw_self_similarity_features(train_data["y"].tolist(),10)
            #params.extend(list(returnable))
            


        d['params'].extend(params)

    except Exception as e:
        d['exception_raised'] = True
        error_detail = error_detail =  str(d['user_id'])+ " "#'error while extracting model features for user : '
        s3_log_error('extract_model_parameters',error_detail+str(e))
        
    
    return d
    



def reduce_data_scope(d):
    """Return a tuple (user_id, {derived model parameters})."""
    row_data = d['params']   
    return [
        (
            row_data[i]
        ) for i in range(0,len(row_data))
    ]

def log_users_with_less_data(d):
    if not len(d['data']) > 9:
        error_messege = str(d['user_id']) + " " + "has played less than 10 days in the given time frame"
        s3_log_error('users_with_less_data',error_messege)
        return False    
    else: 
        return True
    



# # Run Spark

# In[ ]:
sc.setLogLevel("ERROR")

# Retrieve data from s3 csv datastore
df = retrieve_data()
df = df.groupBy('user_id')
select_list = feature.copy()
select_list.append('time_stamp')
df = df.agg(collect_list(struct(select_list)).alias('data'))

df1 = (df.rdd.map(lambda r: transform_data(r))
      .map(lambda d: partition_data(d))
      .filter(lambda d: log_users_with_less_data(d) and d['exception_raised'] == False and d['user_id'] != None)
      .map(lambda d: create_model(d))
      .filter(lambda d: d['exception_raised'] == False)
      .map(lambda d: train_model(d))
      .filter(lambda d: d['exception_raised'] == False)
      .map(lambda d: extract_model_parameters(d))
      .filter(lambda d: d['exception_raised'] == False)
      .map(lambda d: reduce_data_scope(d))
      )


try:
    df2 = spark.createDataFrame(df1)#,schema
    if(not df2.rdd.isEmpty()):
        df2.write.options(header=True).csv(target_dir, mode='append')
except Exception as e:
    error_detail = 'error while writing model features in s3 : '
    s3_log_error('Creating Spark Dataframe and saving in s3',error_detail+str(e))
        
    
spark.stop()

