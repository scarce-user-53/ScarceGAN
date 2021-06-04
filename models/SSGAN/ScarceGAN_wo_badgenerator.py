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

class Scarce_GAN_Normal_Generator:
    
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
        self.discriminator_supervised.compile(optimizer= optimizer_supervised_discriminator,loss=self.custom_loss_supervised_discriminator_v2,metrics=['accuracy'])
        
        self.discriminator_unsupervised = self.build_discriminator_unsupervised()
        self.discriminator_unsupervised.compile(optimizer = optimizer_gan,loss=self.custom_loss_unsupervised_discriminator_v2,metrics=['accuracy'])

        
        self.generator = self.build_generator()
        self.gan = self.build_gan()
        self.gan.compile(optimizer=optimizer_gan,loss=self.custom_loss_unsupervised_discriminator_v2,metrics=['mean_squared_error'])
        
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
        model.add(Dense(3,activation="softmax",kernel_initializer=initializers.glorot_uniform()))
        return model
    

    def build_discriminator_supervised(self):
        model = Sequential()
        model.add(self.discriminator)
        model.add(Dense(self.num_classes + 1, activation="softmax",kernel_initializer=initializers.glorot_uniform()))
        return model


    def build_gan(self):
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator_unsupervised)
        return model
    
    @tf.function
    def custom_loss_supervised_discriminator_v2(self,y_labels,y_pred):
        
        cce = tf.keras.losses.CategoricalCrossentropy()

        mask_l = tf.cast(K.max(y_labels[:,0:4],1) == 1,tf.float32)
        mask_l_r = tf.cast(K.max(y_labels[:,3:4],1) == 1,tf.float32)
        mask_ul = tf.cast(K.max(y_labels[:,4:],1) == 1,tf.float32)
        mask_alpha = tf.math.multiply(self.mask_alpha_1, self.weight_1)
        
        mask_nan = tf.cast(y_pred < 10**-6,tf.float32)
        predict_nan = y_pred + mask_nan
        
        y_labels_l = tf.identity(y_labels)
        y_labels_l = tf.math.multiply(y_labels_l, self.weight_2)
        loss_l = cce(y_labels_l,y_pred) 
        
        mask_risky_fp = tf.cast((tf.cast(K.argmax(y_pred,axis = 1) == 3,tf.float32) - mask_l_r) == 1,tf.float32)
        pu_pr = y_labels[:,4:] / predict_nan[:,3:4]
        mask_nan_pu_pr = tf.cast(pu_pr < 10**-6,tf.float32)
        pu_pr_nan = mask_nan_pu_pr + pu_pr
        pull_entropy = -1 * K.sum((pu_pr * K.log(pu_pr_nan) * mask_risky_fp))/ (K.sum(mask_risky_fp) + 1)

        factor = ( 1 - mask_l_r ) * (1-alpha) * (1-K.max(mask_ul))
        y_labels_uk = K.transpose(tf.math.multiply(K.transpose(tf.math.multiply(self.y_labels_uk_1, self.negate_labeled_classes)),factor))
        loss_uk = cce(y_labels_uk,y_pred) #-1 * K.sum(y_labels_uk * K.log(predict_nan)) 
        
        tot_pred = K.sum(tf.cast(K.argmax(y_pred,axis = 1) == 3,tf.float32)) + 1
        risky_inacc = K.sum(tf.cast((tf.cast(K.argmax(y_pred,axis = 1) == 3,tf.float32) - mask_l_r) == 1,tf.float32)) / tot_pred
        
        return loss_l + loss_uk + risky_inacc  - (1-alpha)*pull_entropy #pull_entropy_l1 + pull_entropy_g1
    
    @tf.function
    def custom_loss_unsupervised_discriminator_v2(self,y_labels,y_pred):
        cce = tf.keras.losses.CategoricalCrossentropy()
        mask_r = tf.cast(K.max(y_labels[:,1:2],1) == 1,tf.float32)
        mask_f = tf.cast(K.max(y_labels[:,2:],1) == 1,tf.float32)
        
        label_f_1 = tf.math.multiply(self.create_unsup_label,self.weight_u9)
        label_f = K.transpose(tf.math.multiply(K.transpose(label_f_1),mask_f))
        cce_f = cce(label_f,y_pred)
        
        ## Categorical Cross entropy for the unlabeled or the real samples 
        mask_nan = tf.cast(y_pred < 10**-6,tf.float32)
        predict_nan = y_pred + mask_nan
        
        #Real sample as Healthy - weight(alpha)
        label_ul_d_1 = tf.math.multiply(self.create_unsup_label,self.weight_u10)
        label_ul_d = K.transpose(tf.math.multiply(K.transpose(label_ul_d_1),mask_r))
        loss_ul_d = cce(label_ul_d,y_pred) 
        
        #Real sample as Unknown - weight(1-alpha)
        label_ul_u_1 = tf.math.multiply(self.create_unsup_label,self.weight_u11)
        label_ul_u = K.transpose(tf.math.multiply(K.transpose(label_ul_u_1),mask_r))
        loss_ul_u = cce(label_ul_u,y_pred) 
        
        cce_u = (loss_ul_d +loss_ul_u)
        
        return cce_u + cce_f
    
    
    def save_weights_toreproduce(self,parent_directory):
        self.generator.save_weights(parent_directory+'/'+'Generator_V5'+run_instance+'.h5')
        self.discriminator.save_weights(parent_directory+'/'+'Discriminator_common_V5'+run_instance+'.h5')
        self.discriminator_supervised.save_weights(parent_directory+'/'+'Discriminator_supervised_V5'+run_instance+'.h5')
        self.discriminator_unsupervised.save_weights(parent_directory+'/'+'Discriminator_unsupervised_V5'+run_instance+'.h5')
        
    
    def save_weights_tracking(self):
        self.generator.save_weights('Model_Weights_SSGAN_Normal_Gen/Model_Trained/'+'Tracking/Generator_V5'+run_instance+'.h5')
        self.discriminator.save_weights('Model_Weights_SSGAN_Normal_Gen/Model_Trained/'+'Tracking/Discriminator_common_V5'+run_instance+'.h5')
        self.discriminator_supervised.save_weights('Model_Weights_SSGAN_Normal_Gen/Model_Trained/'+'Tracking/Discriminator_supervised_V5'+run_instance+'.h5')
        self.discriminator_unsupervised.save_weights('Model_Weights_SSGAN_Normal_Gen/Model_Trained/'+'Tracking/Discriminator_unsupervised_V5'+run_instance+'.h5')
        
   