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
sys.path.append('../')
from AAE_utils import *
s3 = boto3.resource('s3')
bucket = s3.Bucket('bucket_name')

original_dim = 50
input_shape = (original_dim, )
intermediate_dim =15
batch_size = 32
latent_dim = 10
latent_repr_1 = 30
latent_repr_2 = 50
epochs = 100



def sample_z(args):
    mu, log_var = args
    batch = K.shape(mu)[0]
    eps = K.random_normal(shape=(batch, latent_dim), mean=0., stddev=1.)
    return mu + K.exp(log_var / 2) * eps

def build_user_representation(latent_repr_1, input_shape,latent_repr_2):
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(latent_repr_1, activation='sigmoid',kernel_initializer=initializers.glorot_uniform())(inputs)
    x = BatchNormalization()(x)
    x = Dense(latent_repr_2,name='user_latent_representation',kernel_initializer=initializers.glorot_uniform())(x)
    return Model(inputs, x)

def build_encoder(latent_dim, latent_repr_2, intermediate_dim):
    deterministic = 0
    inputs = Input(shape=(latent_repr_2,), name='encoder_input')
    x = Dense(intermediate_dim, activation='relu',kernel_initializer=initializers.glorot_uniform())(inputs)
    x = BatchNormalization()(x)
    if deterministic:
        z = Dense(latent_dim)(x)
        z = BatchNormalization()(z)
    else:
        z_mean = Dense(latent_dim, name='z_mean',kernel_initializer=initializers.glorot_uniform())(x)
        z_mean = BatchNormalization()(z_mean)
        z_log_var = Dense(latent_dim, name='z_log_var',kernel_initializer=initializers.glorot_uniform())(x)
        z_log_var = BatchNormalization()(z_log_var)
        z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    return Model(inputs, z)

def build_discriminator(latent_dim):
    model = Sequential()
    model.add(Dense(15, input_dim=latent_dim, kernel_initializer=initializers.glorot_uniform()))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(21,kernel_initializer=initializers.glorot_uniform()))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation="sigmoid",kernel_initializer=initializers.glorot_uniform()))
    encoded_repr = Input(shape=(latent_dim, ))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)

def build_decoder(latent_dim, intermediate_dim, original_dim):
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu',kernel_initializer=initializers.glorot_uniform())(latent_inputs)
    outputs = Dense(original_dim, activation='relu')(x)
    return Model(latent_inputs, outputs)


optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator
discriminator = build_discriminator(latent_dim)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

counter2vector = build_user_representation(latent_repr_1, input_shape,latent_repr_2)
encoder = build_encoder(latent_dim, latent_repr_2, intermediate_dim)
decoder = build_decoder(latent_dim, intermediate_dim, original_dim)

img = Input(shape=input_shape)
counter2vector_repr = counter2vector(img)
encoded_repr = encoder(counter2vector_repr)
reconstructed_img = decoder(encoded_repr)


discriminator.trainable = False


validity = discriminator(encoded_repr)
weights = np.ones((1,original_dim), dtype=int)
new_weights = K.cast_to_floatx(weights)

def custom_loss(img,reconstructed_img):
    def weighted_mse(y_true,y_pred):
        error_vector = y_true-y_pred
        error_square = K.square(error_vector)
        v1 = K.dot(error_square,K.transpose(new_weights))
        result = v1/(np.sum(new_weights))
        return(result)
    
    def cosine_similarity(y_true,y_pred):
        idx_1 = np.random.randint(0, 50, batch_size)
        risky_users = x_risky[idx_1]
        latent_repr_risky = counter2vector.predict(risky_users)

        idx_2 = np.random.randint(0, x_valid.shape[0], batch_size)
        healthy_users = x_valid[idx_2]
        latent_repr_valid = counter2vector.predict(healthy_users)
        cosine_loss = keras.losses.CosineSimilarity(axis=1)
        return K.mean(cosine_loss(latent_repr_risky, y_true)) - K.mean(cosine_loss(latent_repr_valid, y_true))
  
    def total_loss(y_true,y_pred):     
        reconstruction_loss = weighted_mse(y_true,y_pred)
        
        total_loss =  reconstruction_loss + cosine_similarity(y_true,y_pred)
        return total_loss
    
    return (total_loss)

# The adversarial_autoencoder model  (stacked generator and discriminator)
adversarial_autoencoder = Model(img, [reconstructed_img, validity])

adversarial_autoencoder.compile(loss=[custom_loss(counter2vector_repr,reconstructed_img), 'binary_crossentropy'], loss_weights=[0.5, 0.5], optimizer=optimizer)


