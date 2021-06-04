import sys
sys.path.append('../')
from configuration import *
from utils import *
from models.SSGAN.VanillaSSGAN import *
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


warnings.filterwarnings('ignore')
s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)



class Train_Vanilla_SSGAN:
    
    def __init__(self,x_train,y_train,unlabeled_data,num_class,batch,epoch):
        
        self.supervised_losses = []
        self.iteration_checkpoints = []
        self.accuracies = []
        self.val_losses = []
        self.entropy_class0 = []
        self.entropy_class1 = []
        self.entropy_class2 = []
        self.entropy_class3 = []
        self.precision_class3 = []
        self.recall_class3 = []
        self.best_val_loss = 9999
        self.num_classes = num_class
        self.batch_size = batch
        self.iterations = epoch
        self.sample_interval = sample_interval
        self.z_dim = z_dim
        
        self.unlabeled_data = unlabeled_data
        
        data = pd.DataFrame(x_train)
        data['class'] = list(y_train)
        self.vaild_idx = random.sample(list(range(0,data.shape[0])) , 100)
        valid_condition = data.index.isin(self.vaild_idx)
        
        self.x_train = np.array(data[~valid_condition].iloc[:,:-1])
        self.y_train = np.array(data[~valid_condition].iloc[:,-1:])
        print("Training samples ",self.x_train.shape , self.y_train.shape)


        self.x_valid = x_train[self.vaild_idx]
        self.y_valid = y_train[self.vaild_idx]
        self.y_valid = to_categorical(self.y_valid,num_classes=self.num_classes)
        print("Validation samples ", self.x_valid.shape, self.y_valid.shape)
        
        

        self.ssgan = Vanilla_SSGAN()
               

    
    def batch_labeled(self):
        sample_per_class = int(self.batch_size / self.num_classes)

        for i in range(self.num_classes):
            idx = np.asarray(np.where(self.y_train == i))[0]
            list_ = list(idx)

            pos = random.sample(list_ , sample_per_class)

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
    
    def calculate_entropy(self):
        ssgan_train_predict = -1 * np.log(np.array(self.ssgan.discriminator_supervised.predict(self.x_train)))
        actual_class = np.array(to_categorical(self.y_train,num_classes=self.num_classes))
        self.entropy_class0.append(np.dot(ssgan_train_predict[:,0],actual_class[:,0]))
        self.entropy_class1.append(np.dot(ssgan_train_predict[:,1],actual_class[:,1]))
        self.entropy_class2.append(np.dot(ssgan_train_predict[:,2],actual_class[:,2]))
        self.entropy_class3.append(np.dot(ssgan_train_predict[:,3],actual_class[:,3]))
    
    
    def train(self):
        real = np.ones((self.batch_size,1))
        fake = np.zeros((self.batch_size,1))
        
        for iteration in range(iterations):
            data,labels = self.batch_labeled()
            labels = to_categorical(labels,num_classes=self.num_classes)
            unlabeled_data_batch = self.batch_unlabeled()
            self.ssgan.gan.layers[1].trainable = True
            
            z = np.random.normal(0,1,(self.batch_size,self.z_dim))
            fake_data = self.ssgan.generator.predict(z)
            
            d_supervised_loss,accuracy = self.ssgan.discriminator_supervised.train_on_batch(data,labels)
            d_unsupervised_loss_real,accuracy_real = self.ssgan.discriminator_unsupervised.train_on_batch(unlabeled_data_batch,real)
            d_unsupervised_loss_fake,accuracy_fake = self.ssgan.discriminator_unsupervised.train_on_batch(fake_data,fake)
            d_unsupervised_loss = (d_unsupervised_loss_real + d_unsupervised_loss_fake)/2

            z = np.random.normal(0,1,(self.batch_size,self.z_dim))
            fake_data = self.ssgan.generator.predict(z)
            self.ssgan.gan.layers[1].trainable = False
            generator_loss = self.ssgan.gan.train_on_batch(z,real)
            
            self.supervised_losses.append(d_supervised_loss) 
            self.accuracies.append(100*accuracy)# Training Accuracy
            self.iteration_checkpoints.append(iteration+1)
            val_loss = self.ssgan.discriminator_supervised.evaluate(x=self.x_valid,y=self.y_valid,verbose=0,callbacks = None)
            self.val_losses.append(val_loss[0])
            self.calculate_entropy()
            
            val_predict_class = self.ssgan.discriminator_supervised.predict_classes(self.x_valid)
            val_actual_class = np.dot(self.y_valid,np.array([0,1,2,3]).reshape(4,1)).reshape(self.y_valid.shape[0])
            val_cm = metrics.confusion_matrix(val_actual_class, val_predict_class, labels=[0,1,2,3])
            
            self.precision_class3.append(val_cm[-1,-1]/np.sum(val_cm[:,-1]))
            self.recall_class3.append(val_cm[-1,-1]/np.sum(val_cm[-1,:]))
            
            if val_loss[0] < self.best_val_loss:
                self.best_val_loss = val_loss[0]
                self.ssgan.discriminator_supervised.save_weights('Model_Weights/Vanilla_Supervised_Discriminator.h5') 
            
            if (iteration+1) % sample_interval ==0:
                print("Iteration:",iteration+1,end=",")
                print("Discriminator Supervised Loss:",d_supervised_loss,end=',')
                print('Generator Loss:',generator_loss,end=",")
                print('Accuracy Supervised:',100*accuracy)
                print("\n")   
                

                
                
