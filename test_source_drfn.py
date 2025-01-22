
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd 
import tensorflow as tf
import keras
from keras import ops
from keras import layers
from drfn import DRFN_source
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
dfm3 = pd.read_csv(csv_file_path).pipe(utilities.clip_power)
dfs = dfm3.pipe(utilities.interpolate).pipe(utilities.dt_column).pipe(utilities.normalize_cnt)
dfs_scaling_factor = dfm3.power.max()

# ratioto split train/test and validation

test_percent = 0.30
validation_percent = 0.50

no_test_obs =  int(np.round(test_percent*len(dfs)))
training_data = dfs[:-no_test_obs]
testing = dfs[-no_test_obs:]

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


vae_loaded = keras.models.load_model('drfn_source.keras')


target = []
input = []


for data in test_windowed:
    i,t = data
    target.append(tf.squeeze(t).numpy())
    input.append(tf.squeeze(i[..., :1]).numpy()[0])
    
reconstruction, forecast = vae_loaded.predict(test_windowed)
prediction = (tf.squeeze(forecast).numpy())

reshaped_reconstruction = (tf.squeeze(reconstruction).numpy())



print(reshaped_reconstruction.shape)


utilities.compute_error_metrics(np.array(target)*dfs_scaling_factor,prediction*dfs_scaling_factor)
utilities.plot_comparision(target,prediction)
utilities.plot_comparision(np.array(input)*dfs_scaling_factor,reshaped_reconstruction[:,0]*dfs_scaling_factor)



