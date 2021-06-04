# !pip install keras
# !pip install boto3
# !pip install tensorflow

import sys
sys.path.append('../')
from configuration import *
from utils import *

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


import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings('ignore')

s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)


class Scarce_GAN:
    
    def __init__(self):

        self.z_dim = 5
        self.iterations = iterations
        self.batch_size = batch_size
        self.sample_interval = 500
        self.num_classes = num_classes
        self.original_dim =original_dim
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
        self.gan.compile(optimizer=optimizer_gan,loss=self.custom_loss_generator_fm_sota,metrics=['mean_squared_error'])
        
        self.mask_alpha_1 = K.variable(np.ones((self.batch_size,self.num_classes+1)))
        self.y_labels_uk_1 = K.variable(np.ones((self.batch_size,self.num_classes+1)))
        self.negate_labeled_classes =  K.variable(np.array([0,0,0,1]))
        self.weight_1 = K.variable(np.array([0.7,0.7,0.7,0.3]))
        self.weight_2 = K.variable(np.array([0.7,0.7,1,0]))
        self.create_unsup_label = K.variable(np.ones((self.batch_size,3)))
        self.weight_u9 = K.variable(np.array([0,0,alpha]))
        self.weight_u10 = K.variable(np.array([alpha,0,0]))
        self.weight_u11 = K.variable(np.array([0,1-alpha,0]))
        
        

    def build_generator(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.z_dim, kernel_initializer=initializers.glorot_uniform()))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.original_dim, activation='tanh'))
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
        model.add(Dense(3,activation="sigmoid",kernel_initializer=initializers.glorot_uniform()))
        return model
    

    def build_discriminator_supervised(self):
        model = Sequential()
        model.add(self.discriminator)
        model.add(Dense(4, activation="softmax",kernel_initializer=initializers.glorot_uniform()))
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
    def custom_loss_supervised_discriminator_v2(self,y_labels,y_pred):
       
        cce = tf.keras.losses.CategoricalCrossentropy()

        mask_l = tf.cast(K.max(y_labels[:,0:3],1) == 1,tf.float32)
        mask_l_r = tf.cast(K.max(y_labels[:,2:3],1) == 1,tf.float32)
        mask_ul = tf.cast(K.max(y_labels[:,3:],1) == 1,tf.float32)
        mask_alpha = tf.math.multiply(self.mask_alpha_1, self.weight_1)
        
        mask_nan = tf.cast(y_pred < 10**-6,tf.float32)
        predict_nan = y_pred + mask_nan
        
        y_labels_l = tf.identity(y_labels)
        y_labels_l = tf.math.multiply(y_labels_l, self.weight_2)
        loss_l = cce(y_labels_l,y_pred) 
        
        mask_risky_fp = tf.cast((tf.cast(K.argmax(y_pred,axis = 1) == 2,tf.float32) - mask_l_r) == 1,tf.float32)
        pu_pr = y_labels[:,3:] / predict_nan[:,2:3]
        mask_nan_pu_pr = tf.cast(pu_pr < 10**-6,tf.float32)
        pu_pr_nan = mask_nan_pu_pr + pu_pr
        pull_entropy = -1 * K.sum((pu_pr * K.log(pu_pr_nan) * mask_risky_fp))/ (K.sum(mask_risky_fp) + 1)
        

        factor = ( 1 - mask_l_r ) * (1-alpha) * (1-K.max(mask_ul))
        y_labels_uk = K.transpose(tf.math.multiply(K.transpose(tf.math.multiply(self.y_labels_uk_1, self.negate_labeled_classes)),factor))
        loss_uk = cce(y_labels_uk,y_pred)  
        
        tot_pred = K.sum(tf.cast(K.argmax(y_pred,axis = 1) == 2,tf.float32)) + 1
        risky_inacc = K.sum(tf.cast((tf.cast(K.argmax(y_pred,axis = 1) == 2,tf.float32) - mask_l_r) == 1,tf.float32)) / tot_pred
        
        
        return loss_l + loss_uk + risky_inacc  - (1-alpha)*pull_entropy 
    
    @tf.function
    def custom_loss_unsupervised_discriminator_v2(self,y_labels,y_pred):
        #self.create_unsup_label = K.variable(np.ones((self.batch_size,3)))
        #self.weight_u9 = K.variable(np.array([0,0,1]))
        #self.weight_u10 = K.variable(np.array([alpha,0,0]))
        #self.weight_u11 = K.variable(np.array([0,1-alpha,0]))
        
        #Healthy Unknown Fake
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
        loss_ul_d = cce(label_ul_d,y_pred) #-1 * K.sum(label_ul_d * K.log(predict_nan)) 
        
        #Real sample as Unknown - weight(1-alpha)
        label_ul_u_1 = tf.math.multiply(self.create_unsup_label,self.weight_u11)
        label_ul_u = K.transpose(tf.math.multiply(K.transpose(label_ul_u_1),mask_r))
        loss_ul_u = cce(label_ul_u,y_pred) #-1 * K.sum(label_ul_u * K.log(predict_nan))
        
        cce_u = (loss_ul_d +loss_ul_u)
        
        return cce_u + cce_f
        

        
class Train_Scarce_GAN:
    
    def __init__(self,x_train,y_train,unlabeled_data,num_class,batch,epoch):
        
        self.supervised_losses = []
        self.iteration_checkpoints = []
        self.accuracies = []
        self.val_losses = []
        self.entropy_class0 = []
        self.entropy_class1 = []
        self.entropy_class2 = []

        self.precision_class3 = []
        self.recall_class3 = []
        self.best_val_loss = 9999
        self.num_classes = num_class
        self.batch_size = batch
        self.iterations = iterations
        self.sample_interval = sample_interval
        self.z_dim = 5
        
        self.unlabeled_data = unlabled_data_x
        
        self.x_train = x_train#np.array(data[~valid_condition].iloc[:,:-1])
        self.y_train = y_train#np.array(data[~valid_condition].iloc[:,-1:])
        print("Training samples ",self.x_train.shape , self.y_train.shape)

        self.ssgan = Scarce_GAN()
               
    
    
    def batch_labeled(self):
        sample_per_class = [22,22,20] #[50,50,28]

        for i in range(self.num_classes):
            idx = np.asarray(np.where(self.y_train == i))[0]
            list_ = list(idx)

            pos = random.sample(list_ , sample_per_class[i])

            x_classes= self.x_train[pos]
            y_classes = self.y_train[[pos]]

            if i == 0:
                x_supervised = x_classes
                y_supervised = y_classes

            else:
                x_supervised = np.concatenate((x_supervised, x_classes), axis=0)
                y_supervised = np.concatenate((y_supervised, y_classes), axis=0)


        x_supervised = np.asarray(x_supervised)
        
        return np.array(x_supervised),np.array(y_supervised)


    def batch_unlabeled(self):
        return pd.DataFrame(self.unlabeled_data).sample(self.batch_size)
    
    def stratified_sampling(self,data,labels,num_class):
        #sample_per_class = int(self.batch_size / (num_class))
        sample_per_class = [22,22,20]#[50,50,28]
        for i in range(num_class):
            idx = np.asarray(np.where(labels == i))[0]
            list_ = list(idx)

            pos = random.sample(list_ , sample_per_class)

            x_classes= data[pos]
            y_classes = labels[[pos]]

            if i == 0:
                x_data = x_classes
                y_data = y_classes

            else:
                x_data = np.concatenate((x_data, x_classes), axis=0)
                y_data = np.concatenate((y_data, y_classes), axis=0)


        x_data = np.asarray(x_data)
        
        return np.array(x_data),to_categorical(np.array(y_data),num_classes=num_class)
        

    def calculate_entropy(self):
        ssgan_train_predict = -1 * np.log(np.array(self.ssgan.discriminator_supervised.predict(self.x_train)))
        actual_class = np.array(to_categorical(self.y_train,num_classes=self.num_classes+1))
        self.entropy_class0.append(np.dot(ssgan_train_predict[:,0],actual_class[:,0]))
        self.entropy_class1.append(np.dot(ssgan_train_predict[:,1],actual_class[:,1]))
        self.entropy_class2.append(np.dot(ssgan_train_predict[:,2],actual_class[:,2]))
        
    
    
    
    
    def train(self):
        real = np.ones((self.batch_size,1))
        fake = np.zeros((self.batch_size,1))
        
        for iteration in range(iterations):
            data,labels = self.batch_labeled()
            labels = to_categorical(labels,num_classes=self.num_classes)
            labels = np.hstack((labels,np.zeros((self.batch_size,1))))
            unlabeled_data_batch = self.batch_unlabeled()
            self.ssgan.gan.layers[1].trainable = True
            
            z = np.random.normal(0,1,(self.batch_size,self.z_dim))
            fake_data = self.ssgan.generator.predict(z)

            d_supervised_loss,accuracy = self.ssgan.discriminator_supervised.train_on_batch(data,labels)
            
            
            unsupervised_labels = np.zeros((self.batch_size,3))
            unsupervised_labels_real = unsupervised_labels.copy()
            unsupervised_labels_fake = unsupervised_labels.copy()
            
            unsupervised_labels_real[:,-2] = np.ones((self.batch_size,))
            unsupervised_labels_fake[:,-1] = np.ones((self.batch_size,))
            
            d_unsupervised_loss_real,accuracy_real = self.ssgan.discriminator_unsupervised.train_on_batch(unlabeled_data_batch,unsupervised_labels_real)
            d_unsupervised_loss_fake,accuracy_fake = self.ssgan.discriminator_unsupervised.train_on_batch(fake_data,unsupervised_labels_fake)
            d_unsupervised_loss = (d_unsupervised_loss_real + d_unsupervised_loss_fake)/2

            z = np.random.normal(0,1,(self.batch_size,self.z_dim))
            fake_data = self.ssgan.generator.predict(z)
            self.ssgan.gan.layers[1].trainable = False
            generator_loss = self.ssgan.gan.train_on_batch(z,self.ssgan.discriminator_supervised.layers[0](data))
            
            self.supervised_losses.append(d_supervised_loss) 
            self.accuracies.append(100*accuracy)
            self.iteration_checkpoints.append(iteration+1)
            self.calculate_entropy()

            
            if (iteration+1) % sample_interval ==0:
                print("Iteration:",iteration+1,end=",")
                print("Discriminator Supervised Loss:",d_supervised_loss,end=',')
                print("Discriminator UnSupervised Loss:",d_unsupervised_loss,end=',')
                print('Generator Loss:',generator_loss,end=",")
                print('Accuracy Supervised:',100*accuracy)
                print("\n")


df = pd.read_csv("Data/kddcup.data_10_percent_corrected", sep=",", names=kdd_columns, dtype=kdd_dtypes, index_col=None)
test_set  = pd.read_csv("Data/corrected", sep=",", names=kdd_columns, index_col=None)

# # df = df[~df.duplicated()]
# # test_set = test_set[~test_set.duplicated()]

# df.reset_index(inplace = True)
# test_set.reset_index(inplace = True)


intrusion_r2l = ['warezclient.','warezmaster.','spy.','phf.','multihop.','ftp_write.','guess_passwd.', 'imap.']
intrusion_u2r = ['rootkit.','perl.','loadmodule.','buffer_overflow.']
intrusion_dos = ['land.','smurf.','teardrop.','pod.','neptune.','back.']
intrusion_probe = ['ipsweep.','satan.','portsweep.','nmap.']

def anotate(row):
    if row in intrusion_dos:
        return 0
    elif row in intrusion_probe:
        return 1
    elif row in intrusion_r2l:
        return 1
    elif row in intrusion_u2r:
        return 1
    elif row == 'normal.':
        return 2
    else:
        return -1

# Feature Selection
new_features=['dst_bytes',
 'logged_in',
 'count',
 'srv_count',
 'dst_host_count',
 'dst_host_srv_count',
 'dst_host_same_srv_rate',
 'dst_host_same_src_port_rate','label']

df.label=df.label.apply(lambda x: anotate(x))
df=df[new_features]

print(df.shape)

#label encoding 
for column in df.columns:
    if df[column].dtype == np.object:
        encoded = LabelEncoder()
        
        encoded.fit(df[column])
        df[column] = encoded.transform(df[column])

def stratified_sampling(data,labels,num_class,sample_per_class):
    for i in range(num_class):
        idx = np.asarray(np.where(labels == i))[0]
        list_ = list(idx)

        pos = random.sample(list_ , sample_per_class[i])

        x_classes= data[pos]
        y_classes = labels[[pos]]

        if i == 0:
            x_data = x_classes
            y_data = y_classes

        else:
            x_data = np.concatenate((x_data, x_classes), axis=0)
            y_data = np.concatenate((y_data, y_classes), axis=0)


    x_data = np.asarray(x_data)

    return np.array(x_data),np.array(y_data)
        

labled_data_x, labled_data_y =  stratified_sampling(np.array(df.iloc[:,:-1]),np.array(df.iloc[:,-1]),3,[500,150,500])

print('labled_data_x',labled_data_x.shape)
print('labled_data_y',labled_data_y.shape)


unlabled_data_x,unlabled_data_y =stratified_sampling(np.array(df.iloc[:,:-1]),np.array(df.iloc[:,-1]),3,[4000,176,30000])

print('unlabled_data_x',unlabled_data_x.shape)
print('unlabled_data_y',unlabled_data_y.shape)

alpha = 0.6
iterations = 500
sample_interval = 100
batch_size = 64
original_dim = unlabled_data_x.shape[1]
num_classes = 3


train_network = Train_Scarce_GAN(labled_data_x,labled_data_y,unlabled_data_x,num_classes,batch_size,iterations)
train_network.train()

test_set.label=test_set.label.apply(lambda x: anotate(x))
test_set = test_set[new_features]
test_set = test_set[(test_set['label'] > -1)]

predicted_class = train_network.ssgan.discriminator_supervised.predict_classes(test_set.iloc[:,:-1])
actual_class = test_set.iloc[:,-1]

print(metrics.confusion_matrix(actual_class, predicted_class, labels=[0,1,2,3]))
print(metrics.classification_report(actual_class, predicted_class, labels=[0,1,2,3]))
