import sys
sys.path.append('../')
from configuration import *
from utils import *
from data_collection import *

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
import tensorflow as tf


warnings.filterwarnings('ignore')
s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)
       
class Vanilla_SSGAN:
    
    def __init__(self):

        self.z_dim = z_dim
        self.iterations = iterations
        self.batch_size = batch_size
        self.sample_interval = sample_interval
        self.num_classes = num_classes
        self.original_dim = original_dim
        self.input_shape = (self.original_dim, )
        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(0.001,decay_steps=500,decay_rate=0.96,staircase=True)
        #optimizer_supervised_discriminator = Adam(learning_rate=lr_schedule, beta_1=0.5, beta_2=0.9)
        optimizer_supervised_discriminator = Adam(learning_rate=0.01, beta_1=0.5)
        optimizer_gan = Adam(learning_rate=0.01, beta_1=0.5)
        
        self.discriminator = self.build_discriminator()
        
        self.discriminator_supervised = self.build_discriminator_supervised()
        self.discriminator_supervised.compile(optimizer= optimizer_supervised_discriminator,loss="categorical_crossentropy",metrics=['accuracy'])
        
        self.discriminator_unsupervised = self.build_discriminator_unsupervised()
        self.discriminator_unsupervised.compile(optimizer = optimizer_gan,loss='binary_crossentropy',metrics=['accuracy'])
        
        self.generator = self.build_generator()
        self.gan = self.build_gan()
        self.gan.compile(optimizer=optimizer_gan,loss="binary_crossentropy",metrics=['accuracy'])
        
        
    # generator
    def build_generator(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.z_dim, kernel_initializer=initializers.glorot_uniform()))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.original_dim, activation='relu'))
        return model

    # discriminator - latent space 
    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(15, input_dim=self.original_dim, kernel_initializer=initializers.glorot_uniform()))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(21,kernel_initializer=initializers.glorot_uniform()))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(8,kernel_initializer=initializers.glorot_uniform()))
        model.add(LeakyReLU(alpha=0.2))
        encoded_repr = Input(shape=(self.original_dim, ))
        validity = model(encoded_repr)
        return Model(encoded_repr, validity)
    
    # unsupervised discriminator - adversary
    def build_discriminator_unsupervised(self):
        model = Sequential()
        model.add(self.discriminator)
        model.add(Dense(1,activation="sigmoid",kernel_initializer=initializers.glorot_uniform()))
        return model
    
    # supervised discriminator 
    def build_discriminator_supervised(self):
        model = Sequential()
        model.add(self.discriminator)
        model.add(Dense(self.num_classes, activation="softmax",kernel_initializer=initializers.glorot_uniform()))
        return model

    # Generator with unsupervised discriminator as adversary network
    def build_gan(self):
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator_unsupervised)
        return model
    
    def save_weights_toreproduce(self,parent_directory):
        self.generator.save_weights(parent_directory+'/'+'Generator_V5'+run_instance+'.h5')
        self.discriminator.save_weights(parent_directory+'/'+'Discriminator_common_V5'+run_instance+'.h5')
        self.discriminator_supervised.save_weights(parent_directory+'/'+'Discriminator_supervised_V5'+run_instance+'.h5')
        self.discriminator_unsupervised.save_weights(parent_directory+'/'+'Discriminator_unsupervised_V5'+run_instance+'.h5')
        

    
    def save_weights_tracking(self):
        self.generator.save_weights('Model_Weights_Vanilla/Model_Trained/'+'Tracking/Generator_V5'+run_instance+'.h5')
        self.discriminator.save_weights('Model_Weights_Vanilla/Model_Trained/'+'Tracking/Discriminator_common_V5'+run_instance+'.h5')
        self.discriminator_supervised.save_weights('Model_Weights_Vanilla/Model_Trained/'+'Tracking/Discriminator_supervised_V5'+run_instance+'.h5')
        self.discriminator_unsupervised.save_weights('Model_Weights_Vanilla/Model_Trained/'+'Tracking/Discriminator_unsupervised_V5'+run_instance+'.h5')
    
    
