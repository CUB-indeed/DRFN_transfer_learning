from model_utilities import TimeSeriesPersistenceModel
import utilities

import pandas as pd
import numpy as np



# source 

window_len = 5
forecast_len = 1
latent_dim = 4
n_total_features = 10
n_aleatoric_features = 4
n_deterministic_features = n_total_features - n_aleatoric_features
batch_size = 50

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

model_source = TimeSeriesPersistenceModel(training_data,dfs_scaling_factor)
model_source.prepare_model()
model_source.plot_3d()

a,p = utilities.get_persistance_predictions(model_source,training_data,dfs_scaling_factor)

print('persistance_source')
utilities.compute_error_metrics(a,p)

# target 1

dfk = pd.read_csv('/Users/sakshisharma/Desktop/TL_codes/target1.csv')
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


a,p = utilities.get_persistance_predictions(model_source,training_data,dft1_scaling_factor)

print('persistance_target1')
utilities.compute_error_metrics(a,p)

#target2

dfk = pd.read_csv('/Users/sakshisharma/Desktop/TL_codes/target2.csv')
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


a,p = utilities.get_persistance_predictions(model_source,training_data,dft1_scaling_factor)

print('persistance_target2')
utilities.compute_error_metrics(a,p)