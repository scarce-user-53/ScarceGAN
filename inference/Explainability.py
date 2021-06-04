import sys
sys.path.append('../')
from configuration import *
from data_collection import *
from utils import *

from models.SSGAN.ScarceGAN import *
from models.SSGAN.VanillaSSGAN import *
from models.SSGAN.ScarceGAN_wo_badgenerator import *
from models.SSGAN.ScarceGAN_wo_leewayterm import *

from training.SSGAN.Train_ScarceGAN import *
from training.SSGAN.Train_SSGAN_wo_badgenerator import *
from training.SSGAN.Train_SSGAN_wo_leewayterm import *
from training.SSGAN.Train_VanillaSSGAN import *

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


class Explainability:
    
    def __init__(self,trained_ssgan):
        #Plot Train Parameters
        self.trained_ssgan = trained_ssgan
        self.generate_training_plots(trained_ssgan.accuracies,'supervised_sccuracy')
        self.generate_training_plots(trained_ssgan.supervised_losses,'supervised_losses')
        self.generate_training_plots(trained_ssgan.val_losses,'validation_losses')
        self.generate_training_plots(trained_ssgan.entropy_class0,'entropy_dormant')
        self.generate_training_plots(trained_ssgan.entropy_class1,'entropy_normal')
        self.generate_training_plots(trained_ssgan.entropy_class2,'entropy_heavy')
        self.generate_training_plots(trained_ssgan.entropy_class3,'entropy_risky')
        self.generate_training_plots(trained_ssgan.precision_class3,'precision_risky')
        self.generate_training_plots(trained_ssgan.recall_class3,'recall_risky')
        
    
    def generate_training_plots(self,data,data_name):
        epochs = [i for i in range(0,iterations)]
        plt.plot(epochs,data,color='blue', marker='o', linestyle='dashed')
        plt.xlabel("Iterations")
        plt.ylabel(data_name)
        plt.show()
        
    def predict_on_labeled_test_set(self,scaled_test_data,actual_class):
        predict_class = self.trained_ssgan.ssgan.discriminator_supervised.predict_classes(scaled_test_data)
        print(metrics.confusion_matrix(actual_class, predict_class, labels=[0,1,2,3,4]))
        print(metrics.classification_report(actual_class, predict_class, labels=[0,1,2,3,4]))
        return self.trained_ssgan.ssgan.discriminator_supervised.predict_classes(scaled_test_data),self.trained_ssgan.ssgan.discriminator_supervised.predict(scaled_test_data)
        
    def predict_on_unlabeled_test_set(self,scaled_test_data):
        check = pd.DataFrame(self.trained_ssgan.ssgan.discriminator_supervised.predict_classes(scaled_test_data),columns = ["Class"])
        print("Predicted Dormant User:",check[check["Class"] == 0].shape[0])
        print("Predicted Normal User:",check[check["Class"] == 1].shape[0])
        print("Predicted Heavy User:",check[check["Class"] == 2].shape[0])
        print("Predicted Risky User:",check[check["Class"] == 3].shape[0])
        print("Predicted Unknown User:",check[check["Class"] == 4].shape[0])
        return self.trained_ssgan.ssgan.discriminator_supervised.predict(scaled_test_data)
    
    def predict_on_labeled_test_set_best_model(self,scaled_test_data,actual_class,best_model):
        predict_class = best_model.discriminator_supervised.predict_classes(scaled_test_data)
        print(metrics.confusion_matrix(actual_class, predict_class, labels=[0,1,2,3]))
        print(metrics.classification_report(actual_class, predict_class, labels=[0,1,2,3]))
        return best_model.discriminator_supervised.predict_classes(scaled_test_data),best_model.discriminator_supervised.predict(scaled_test_data)
        
    def predict_on_unlabeled_test_set_best_model(self,scaled_test_data,best_model):
        check = pd.DataFrame(best_model.discriminator_supervised.predict_classes(scaled_test_data),columns = ["Class"])
        print("Predicted Dormant User:",check[check["Class"] == 0].shape[0])
        print("Predicted Normal User:",check[check["Class"] == 1].shape[0])
        print("Predicted Heavy User:",check[check["Class"] == 2].shape[0])
        print("Predicted Risky User:",check[check["Class"] == 3].shape[0])
        print("Predicted Unknown User:",check[check["Class"] == 4].shape[0])
        return best_model.discriminator_supervised.predict(scaled_test_data)