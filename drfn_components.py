import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers
from model_utilities import Sampling
import utilities


def create_encoder(n_features, window_len, latent_dim):
    ### Model Inputs ###
    encoder_inputs = tf.keras.Input(shape=(window_len, n_features), name='encoder_inputs')
    # transformer_encoder = keras_hub.layers.TransformerEncoder(intermediate_dim=64, num_heads=8)
    # ### Encoder layers and forward pass ###
    # transformer_out = transformer_encoder(encoder_inputs)
    # First LSTM layer
    x = layers.LSTM(100, activation="tanh", return_sequences=True)(encoder_inputs)
    # Second LSTM layer
    x = layers.LSTM(32, activation="tanh", return_sequences=False)(x)
    
    # Fully connected layer
    x = layers.Dense(16, activation="relu")(x)
    
    # Latent space
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    # Encoder model
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    
    return encoder

def create_decoder(n_features,window_len,latent_dim):
    n_features = n_features
    window_len = window_len
    latent_dim = latent_dim 
    
    # Latent input
    latent_inputs = layers.Input(shape=(latent_dim,))

    x = layers.Dense(100, activation="relu")(latent_inputs)
    x = layers.Dense(50, activation="relu")(x)
    x = layers.Dense(window_len * n_features, activation="relu")(x)
    x = layers.Reshape((window_len, n_features))(x)
 
    x = layers.LSTM(50, return_sequences=True, activation="relu")(x)
    x = layers.LSTM(10, return_sequences=True, activation="relu")(x)
    x = layers.LSTM(5, return_sequences=True, activation="relu")(x)

    
    decoder_outputs = layers.Dense(1, activation="relu")(x) 
 
    decoder = keras.Model(latent_inputs, decoder_outputs, name="reconstruction")
    decoder.summary()

    return decoder

# def create_forecaster(n_features,latent_dim,forecast_len):
    
#     n_features = n_features
#     forecast_len = forecast_len
#     latent_dim = latent_dim 
    
#     # Latent input
#     latent_inputs = layers.Input(shape=(latent_dim,))

#     x = layers.Dense(100, activation="relu")(latent_inputs)
#     x = layers.Dense(50, activation="relu")(x)
#     x = layers.Dense(forecast_len * n_features, activation="relu")(x)
#     x = layers.Reshape((forecast_len, n_features))(x)

#     x = layers.LSTM(50, return_sequences=True, activation="relu")(x)
#     x = layers.LSTM(10, return_sequences=True, activation="relu")(x)
#     x = layers.LSTM(5, return_sequences=True, activation="relu")(x)

    
    
#     forecaster_outputs = layers.Dense(1, activation="relu")(x)  
    
#     forecaster = keras.Model(latent_inputs, forecaster_outputs, name="forecast")
#     forecaster.summary()

#     return forecaster


def create_forecaster(latent_dim, forecast_len):
    latent_inputs = layers.Input(shape=(latent_dim,))

    # Expand latent inputs into sequences directly
    x = layers.Dense(forecast_len * 10, activation="relu")(latent_inputs)
    x = layers.Reshape((forecast_len, 10))(x)

    # LSTM layers for temporal processing
    x = layers.LSTM(50, return_sequences=True, activation="tanh")(x)
    x = layers.LSTM(25, return_sequences=True, activation="tanh")(x)
    
    # Output layer with linear activation
    forecaster_outputs = layers.Dense(1, activation="linear")(x)
    
    # Forecaster model
    forecaster = keras.Model(latent_inputs, forecaster_outputs, name="forecast")
    forecaster.summary()

    return forecaster
    






