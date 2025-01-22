import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import ops
from keras import layers

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon
    
class TimeSeriesPersistenceModel:
    def __init__(self, data, sf):
        """
        Initialize the model with the data.

        Parameters:
        - data: DataFrame with 'datetime' and 'power' columns.
        """
        self.sf = sf
        self.data = data
        self.multi_index_df = None

    def prepare_model(self):
        """
        Prepare the MultiIndex DataFrame with average power for each day of the year and hour of the day.
        """
        # Ensure 'datetime' column is in datetime format
        self.data = self.data.reset_index(names=['datetime'])
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])

        # Extract day of year and hour of day
        self.data['day_of_year'] = self.data['datetime'].dt.dayofyear
        self.data['hour_of_day'] = self.data['datetime'].dt.hour

        # Group by day of year and hour of day, then calculate the average power
        grouped_df = self.data.groupby(['day_of_year', 'hour_of_day'])['power'].mean().reset_index()


        # Create a MultiIndex DataFrame
        self.multi_index_df = grouped_df.set_index(['day_of_year', 'hour_of_day'])

        # Rename the columns for clarity
        self.multi_index_df.rename(columns={'power': 'average_power'}, inplace=True)

    def get_expected_power(self, input_datetime):
        """
        Given a datetime, return the expected power based on the MultiIndex DataFrame.

        Parameters:
        - input_datetime: datetime object for which to predict the power.

        Returns:
        - expected_power: The expected power value.
        """
        if self.multi_index_df is None:
            raise ValueError("Model has not been prepared. Call prepare_model() first.")

        # Ensure input_datetime is a datetime object
        input_datetime = pd.to_datetime(input_datetime)

        # Extract day of year and hour of day
        day_of_year = input_datetime.dayofyear
        hour_of_day = input_datetime.hour

        # Look up the expected power in the multi_index_df
        try:
            expected_power = self.multi_index_df.loc[(day_of_year, hour_of_day)]['average_power']
        except KeyError:
            expected_power = None  # Handle the case where the specific day/hour is not in the data

        return expected_power


    def plot_3d(self):
      X = np.array([x for x in self.multi_index_df.index.get_level_values(0)])
      Y = np.array([x for x in self.multi_index_df.index.get_level_values(1)])
      Z = self.multi_index_df['average_power'].values*self.sf
      fig = plt.figure(figsize=(10,6))
      ax = fig.add_subplot(111, projection='3d')
      surf = ax.plot_trisurf(Y,X,Z, cmap=cm.jet, linewidth=0.2)
      # Customize the z axis.
      ax.set_zlim(0, 1*self.sf)
      ax.zaxis.set_major_locator(LinearLocator(5))
      ax.zaxis.set_major_formatter(FormatStrFormatter('%.f'))
      # Add a color bar which maps values to colors.
      fig.colorbar(surf, shrink=0.5, aspect=5)
      ax.set_xlabel('Hour of day')
      ax.set_ylabel('Day of year')
      ax.set_zlabel('Power Output')
      plt.show()
