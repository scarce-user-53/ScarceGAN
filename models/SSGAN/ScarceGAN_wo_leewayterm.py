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

 
     
class Scarce_GAN_Rem_Unknown_Class:
    
    def __init__(self):

        self.z_dim = z_dim
        self.iterations = iterations
        self.batch_size = batch_size
        self.sample_interval = sample_interval
        self.num_classes = num_classes
        self.original_dim = original_dim
        self.input_shape = (self.original_dim, )
        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(0.001,decay_steps=500,decay_rate=0.96,staircase=True)
        optimizer_supervised_discriminator = Adam(learning_rate=lr_schedule, beta_1=0.5, beta_2=0.9)
        #optimizer_supervised_discriminator = Adam(learning_rate=0.0001, beta_1=0.5)
        optimizer_gan = Adam(learning_rate=0.0001, beta_1=0.5)
        
        self.discriminator = self.build_discriminator()
        
        self.discriminator_supervised = self.build_discriminator_supervised()
        self.discriminator_supervised.compile(optimizer= optimizer_supervised_discriminator,loss="categorical_crossentropy",metrics=['accuracy'])
        
        self.discriminator_unsupervised = self.build_discriminator_unsupervised()
        self.discriminator_unsupervised.compile(optimizer = optimizer_gan,loss='binary_crossentropy',metrics=['accuracy'])
     
        self.generator = self.build_generator()
        self.gan = self.build_gan()
        self.gan.compile(optimizer=optimizer_gan,loss=self.custom_loss_generator_fm_sota,metrics=['mean_squared_error'])
        
        self.mask_alpha_1 = K.variable(np.ones((self.batch_size,self.num_classes+1)))
        self.create_label = K.variable(np.ones((self.batch_size,self.num_classes+2)))
        self.y_labels_uk_1 = K.variable(np.ones((self.batch_size,self.num_classes+1)))
        self.negate_labeled_classes =  K.variable(np.array([0,0,0,0,1]))
        self.weight_1 = K.variable(np.array([alpha,alpha,alpha,alpha,1-alpha]))
        self.weight_2 = K.variable(np.array([alpha,alpha,alpha,alpha,0]))
        self.weight_3 = K.variable(np.array([alpha * beta,0,0,0,0]))
        self.weight_4 = K.variable(np.array([0,alpha * beta,0,0,0]))
        self.weight_5 = K.variable(np.array([0,0,alpha * beta,0,0]))
        self.weight_6 = K.variable(np.array([0,0,0,alpha * (1-beta),0]))
        self.weight_7 = K.variable(np.array([0,0,0,0,1-alpha]))
        self.weight_8 = K.variable(np.array([0,0,0,1,0]))
        self.weight_u3 = K.variable(np.array([alpha * beta,0,0,0,0,0]))
        self.weight_u4 = K.variable(np.array([0,alpha * beta,0,0,0,0]))
        self.weight_u5 = K.variable(np.array([0,0,alpha * beta,0,0,0]))
        self.weight_u6 = K.variable(np.array([0,0,0,alpha * (1-beta),0,0]))
        self.weight_u7 = K.variable(np.array([0,0,0,0,1-alpha,0]))
        self.weight_u8 = K.variable(np.array([0,0,0,0,0,1]))
        
        self.create_unsup_label = K.variable(np.ones((self.batch_size,self.num_classes-1)))
        self.weight_u9 = K.variable(np.array([0,0,1]))
        self.weight_u10 = K.variable(np.array([alpha,0,0]))
        self.weight_u11 = K.variable(np.array([0,1-alpha,0]))

    def build_generator(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.z_dim, kernel_initializer=initializers.glorot_uniform()))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.original_dim, activation='relu'))
        return model


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
    
    def build_discriminator_unsupervised(self):
        model = Sequential()
        model.add(self.discriminator)
        model.add(Dense(1,activation="sigmoid",kernel_initializer=initializers.glorot_uniform()))
        return model
    

    def build_discriminator_supervised(self):
        model = Sequential()
        model.add(self.discriminator)
        model.add(Dense(self.num_classes , activation="softmax",kernel_initializer=initializers.glorot_uniform()))
        return model


    def build_gan(self):
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        return model
    
    @tf.function
    def custom_loss_generator_fm_sota(self,y_labels,y_pred):
        fm_loss = K.mean(K.abs(K.mean(y_pred,axis = 0)- K.mean(y_labels,axis = 0)))
        
        fogoz_cs = K.dot(K.l2_normalize(y_pred),K.transpose(K.l2_normalize(y_pred)))
        mask = K.constant(np.ones((fogoz_cs.shape[0],fogoz_cs.shape[0])) - np.identity(fogoz_cs.shape[0]))
        fogoz_pt_masked = fogoz_cs * mask
        
        pt_loss = K.sum(K.sum(K.square(fogoz_pt_masked),axis=0)/(y_pred.shape[0]*(y_pred.shape[0]-1)))
        
        return fm_loss + pt_loss 
     
    @tf.function
    def custom_loss_generator_fm_pt_lds(self,fogoz,ls_ds):
        predict_real_proba = self.discriminator_unsupervised.layers[1](fogoz)
        fake_sample_mask = tf.cast(predict_real_proba < 0.25, tf.float32)
        
        predict_class_proba = self.discriminator_supervised.layers[1](fogoz)
        mask_lds =  tf.cast(predict_class_proba > epsilon, tf.float32)
        predict_class_proba_eps = predict_class_proba * mask_lds
        
        class_proba = K.max(predict_class_proba, 1)*fake_sample_mask
        mask_nan = tf.cast(class_proba == 0, tf.float32)
        class_proba = class_proba + mask_nan
        lds = K.sum(K.log(class_proba))
        
        fm_loss = K.mean(K.square(K.l2_normalize(K.mean(fogoz,axis = 0)- K.mean(ls_ds,axis = 0))))
    
        fogoz_cs = K.dot(K.l2_normalize(fogoz),K.transpose(K.l2_normalize(fogoz)))
        mask = K.constant(np.ones((fogoz_cs.shape[0],fogoz_cs.shape[0])) - np.identity(fogoz_cs.shape[0]))
        fogoz_pt_masked = fogoz_cs * mask
        pt_loss = K.sum(K.sum(K.square(fogoz_pt_masked),axis=0)/(fogoz.shape[0]*(fogoz.shape[0]-1)))
        
        return fm_loss + pt_loss + lds
    
    
    def save_weights_toreproduce(self,parent_directory):
        self.generator.save_weights(parent_directory+'/'+'Generator_V5'+run_instance+'.h5')
        self.discriminator.save_weights(parent_directory+'/'+'Discriminator_common_V5'+run_instance+'.h5')
        self.discriminator_supervised.save_weights(parent_directory+'/'+'Discriminator_supervised_V5'+run_instance+'.h5')
        self.discriminator_unsupervised.save_weights(parent_directory+'/'+'Discriminator_unsupervised_V5'+run_instance+'.h5')
        
    
    
    def save_weights_tracking(self):
        self.generator.save_weights('Model_Weights_SSGAN_No_Unknown/Model_Trained/'+'Tracking/Generator_V5'+run_instance+'.h5')
        self.discriminator.save_weights('Model_Weights_SSGAN_No_Unknown/Model_Trained/'+'Tracking/Discriminator_common_V5'+run_instance+'.h5')
        self.discriminator_supervised.save_weights('Model_Weights_SSGAN_No_Unknown/Model_Trained/'+'Tracking/Discriminator_supervised_V5'+run_instance+'.h5')
        self.discriminator_unsupervised.save_weights('Model_Weights_SSGAN_No_Unknown/Model_Trained/'+'Tracking/Discriminator_unsupervised_V5'+run_instance+'.h5')
        
        