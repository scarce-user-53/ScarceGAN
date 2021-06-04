from __future__ import print_function, division
from sklearn.externals import joblib
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
from keras.layers import Lambda
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras import initializers
import numpy as np
import pandas as pd
import boto3
import io
import tensorflow as tf
from tensorflow.keras import backend as K
import sys
sys.path.append('/models/AAE')
from AAE_model import *

s3 = boto3.resource('s3')
bucket = s3.Bucket('bucket_name')

epochs=7000
batch_size=32

# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

def sample_prior(latent_dim, batch_size):
    return np.random.normal(size=(batch_size, latent_dim))

d_loss_0=[]
d_loss_1 = []
g_loss_0=[]
g_loss_1 = []
for epoch in range(epochs):

    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random batch of images
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    imgs = x_train[idx]
    latent_repr = counter2vector.predict(imgs)
    latent_fake = encoder.predict(latent_repr)
    
    # Here we generate the "TRUE" samples
    latent_real = sample_prior(latent_dim, batch_size)
                      
    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(latent_real, valid)
    d_loss_fake = discriminator.train_on_batch(latent_fake, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train Generator
    # ---------------------

    # Train the generator
    g_loss = adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])

    # Plot the progress (every 500th epoch)
    if epoch % 500 == 0:
        print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
         
    d_loss_0.append(d_loss[0])
    d_loss_1.append(d_loss[1])
    g_loss_0.append(d_loss[0])
    g_loss_1.append(d_loss[1])
    
    
adversarial_file_name = 'AAE_Model_Weights/waae_adv_explainable_prophet_dtw_v3.h5'
counter2vector_file_name = 'AAE_Model_Weights/waae_counter2vector_explainable_prophet_dtw_v3.h5'
encoder_file_name = 'AAE_Model_Weights/waae_encoder_explainable_prophet_dtw_v3.h5'
decoder_file_name = 'AAE_Model_Weights/waae_decoder_explainable_prophet_dtw_v3.h5'
discriminator_file_name = 'AAE_Model_Weights/waae_disc_explainable_prophet_dtw_v3.h5'

adversarial_autoencoder.save_weights(adversarial_file_name)
counter2vector.save_weights(counter2vector_file_name)
encoder.save_weights(encoder_file_name)
decoder.save_weights(decoder_file_name)
discriminator.save_weights(discriminator_file_name)

