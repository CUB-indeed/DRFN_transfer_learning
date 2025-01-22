
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd 
import tensorflow as tf
import keras
from keras import ops
from keras import layers

from drfn import DRFN_source, DRFN_target
import drfn_components
import utilities
import matplotlib.pyplot as plt




window_len = 12
forecast_len = 12
latent_dim = 4
n_total_features = 18
batch_size = 50


# data prep
csv_file_path = ''
dfk = pd.read_csv(csv_file_path)
dft1 = dfk.pipe(utilities.interpolate).pipe(utilities.dt_column).pipe(utilities.normalize_cnt)
dft1_scaling_factor = dfk.power.max()

# ratioto split train/test and validation

test_percent = 0.30
validation_percent = 0.50

no_test_obs =  int(np.round(test_percent*len(dft1)))
training_data = dft1[:-no_test_obs]
testing = dft1[-no_test_obs:]

# Breaking the testing data into validation and out of sample data
no_validation_obs = int(np.round(validation_percent*len(testing)))
validation_data = testing[:-no_validation_obs]
test_data = testing[-no_validation_obs:]


# Extract training, validation, and test tf.data.Dataset objects



training_windowed = utilities.create_dataset(training_data,
                                   window_len,
                                   forecast_len,
                                   batch_size)


validation_windowed  = utilities.create_dataset(validation_data,
                                     window_len,
                                     forecast_len,
                                     batch_size)

test_windowed  = utilities.create_dataset(test_data,
                               window_len,
                               forecast_len,
                               batch_size=1).batch(1)


encoder = drfn_components.create_encoder(n_total_features,window_len,latent_dim)
decoder = drfn_components.create_decoder(n_total_features,window_len,latent_dim)

drfn_source = keras.models.load_model('drfn_source.keras')

# DRFN prediction without transfer learning. 
target = []
input = []


for data in test_windowed:
    i,t = data
    target.append(tf.squeeze(t).numpy())
    input.append(tf.squeeze(i).numpy())


reconstruction, forecast = drfn_source.predict(test_windowed)

prediction = (tf.squeeze(forecast).numpy())
reshaped_reconstruction = tf.reshape(reconstruction, (-1, reconstruction.shape[2])) 
combined_input = np.concatenate(input, axis=0)

utilities.compute_error_metrics(np.array(target)*dft1_scaling_factor,prediction*dft1_scaling_factor)
utilities.plot_comparision(target,prediction)


# Transfer learning in DRFN

encoder_source = drfn_source.encoder
forecaster_source = drfn_source.forecaster
drfn_target = DRFN_target(encoder_source,encoder, decoder, forecaster_source, forecastor_training=False)
drfn_target.build(input_shape=(None, window_len, n_total_features))
drfn_target.compile(optimizer=keras.optimizers.Adam())
drfn_target.fit(training_windowed,epochs=20)


reconstruction_tl, forecast_tl = drfn_source.predict(test_windowed)
prediction_tl = (tf.squeeze(forecast_tl).numpy())

reshaped_reconstruction = tf.reshape(reconstruction_tl, (-1, reconstruction_tl.shape[2])) 
combined_input = np.concatenate(input, axis=0)

utilities.compute_error_metrics(np.array(target)*dft1_scaling_factor,prediction_tl*dft1_scaling_factor)
utilities.plot_comparision(target,prediction_tl)


# Transfer learning in with labels
encoder_label = drfn_components.create_encoder(n_total_features,window_len,latent_dim)
decoder_label = drfn_components.create_decoder(n_total_features,window_len,latent_dim)

encoder_source = drfn_source.encoder
forecaster_source = drfn_source.forecaster
drfn_target = DRFN_target(encoder_source,encoder_label, decoder_label, forecaster_source, forecastor_training=True)
drfn_target.build(input_shape=(None, window_len, n_total_features))
drfn_target.compile(optimizer=keras.optimizers.Adam())
drfn_target.fit(training_windowed,epochs=100)


reconstruction_tl, forecast_tl = drfn_source.predict(test_windowed)
prediction_tl = (tf.squeeze(forecast_tl).numpy())

reshaped_reconstruction = tf.reshape(reconstruction_tl, (-1, reconstruction_tl.shape[2])) 
combined_input = np.concatenate(input, axis=0)

utilities.compute_error_metrics(np.array(target)*dft1_scaling_factor,prediction_tl*dft1_scaling_factor)
utilities.plot_comparision(target,prediction_tl)
