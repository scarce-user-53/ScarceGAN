!pip install keras
!pip install boto3
!pip install tensorflow

import sys
sys.path.append('../')
from configuration import *
from data_collection import *
from utils import *

from inference.Explainability import *
from models.SSGAN.ScarceGAN import *
from models.SSGAN.VanillaSSGAN import *
from models.SSGAN.ScarceGAN_wo_badgenerator import *
from models.SSGAN.ScarceGAN_wo_leewayterm import *

from training.SSGAN.Train_ScarceGAN import *
from training.SSGAN.Train_SSGAN_wo_badgenerator import *
from training.SSGAN.Train_SSGAN_wo_leewayterm import *
from training.SSGAN.Train_VanillaSSGAN import *


# from __future__ import print_function, division
# from sklearn.externals import joblib
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Lambda
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
# from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from tensorflow.keras import initializers
import numpy as np
import pandas as pd
import boto3
import random
import io
import keras
import warnings 

warnings.filterwarnings('ignore')

s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)

def generate_test_results(trained_network):
    test_set_2= pd.read_csv('Data/Test_Set_2.csv')
    test_set_2_scaled = scaler.transform(test_selecteddata.iloc[:,:-2])

    explainable_1 = Explainability(trained_network)
    print("Test Prediction Report for Test Set 1")
    Test_Class,Test_Probabilities = explainable_1.predict_on_labeled_test_set(scaled_test_data,y_test)
    print("Test Prediction Report for Test Set 2")
    Test_Class,Test_Probabilities = explainable_1.predict_on_labeled_test_set(np.array(test_set_2_scaled),test_set_2.iloc[:,-2:-1])

    production_users_day_1 = read_user_data(s3_location_input_data_production)
    production_users_day_1.fillna(0,inplace = True)
    production_users_day_1 = scaler.transform(production_users_day_1.iloc[:,1:51])
    print(" \n Production Users Report as Test Set:",production_users_day_1.shape[0])
    Test_Probabilities_production = explainable_1.predict_on_unlabeled_test_set(production_users_day_1)
    
    

if __name__ == '__main__':
    x_train = np.concatenate([x_dormant,x_heavy,x_normal,x_risky],axis=0)
    y_train = np.concatenate([y_dormant,y_heavy,y_normal,y_risky],axis=0)

    
    train_network_scarcegan = Train_Scarce_GAN(x_train,y_train,unlabeled_data,4,batch_size,iterations)
    train_network_scarcegan.train()
    print(" \n Scarce GAN Test results")
    generate_test_results(train_network_scarcegan)
    
    train_network_ssgan_wo_badgen = Train_Scarce_GAN_Normal_Generator(x_train,y_train,unlabeled_data,4,batch_size,iterations)
    train_network_ssgan_wo_badgen.train()
    
    train_network_vanilla_ssgan = Train_Vanilla_SSGAN(x_train,y_train,unlabeled_data,4,batch_size,iterations)
    train_network_vanilla_ssgan.train()
    
    train_network_ssgan_wo_leeway = Train_Scarce_Scarce_GAN_Rem_Unknown_Class(x_train,y_train,unlabeled_data,4,batch_size,iterations)
    train_network_ssgan_wo_leeway.train()

    