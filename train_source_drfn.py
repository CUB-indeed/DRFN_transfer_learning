
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
from keras.api.callbacks import ReduceLROnPlateau




window_len = 12
forecast_len = 12
latent_dim = 4
n_total_features = 18
batch_size = 50


# data prep

dfm3 = pd.read_csv('/Users/sakshisharma/Desktop/TL_codes/source.csv').pipe(utilities.clip_power)
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


encoder = drfn_components.create_encoder(n_total_features,window_len,latent_dim)
decoder = drfn_components.create_decoder(n_total_features,window_len,latent_dim)
forecaster = drfn_components.create_forecaster(n_total_features,latent_dim,forecast_len) 

lr = ReduceLROnPlateau(monitor='forecast_loss',factor=0.2, patience=5,cooldown=5, min_lr=0.00000001)


drfn = DRFN_source(encoder, decoder, forecaster)
drfn.build(input_shape=(None, window_len, n_total_features),layer_init="random_normal")
drfn.compile(optimizer=keras.optimizers.Adam())
drfn.fit(training_windowed,
        epochs=100,callbacks=[lr])



# Save the model
drfn.save('drfn_source.keras')