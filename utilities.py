import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2


import tensorflow as tf
import plotly.graph_objects as go


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


from datetime import datetime, timedelta




def interpolate(df):
  return df.interpolate(method='linear')

def dt_column(df):
    df['datetime'] = pd.to_datetime(df['datetime'] )
    df = df.set_index('datetime')
    df['month_of_year'] = df.index.month
    one_hot_months = pd.get_dummies(df['month_of_year'], prefix='month').astype(int)
    df = df[['power','temp','cloudcover','solarradiation', 'Day sin', 'Day cos']]
    result_df = pd.concat([df, one_hot_months], axis=1)
   
    return result_df

def normalize_cnt(df):
    df = df.copy()
    df['power'] = df['power'] / df['power'].max()
    df['temp'] = df['temp'] / df['temp'].max()
    df['cloudcover'] = df['cloudcover'] / df['cloudcover'].max()
    df['solarradiation'] = df['solarradiation'] / df['solarradiation'].max()
    return df

def final_df(df):
  df = df[['power','temp','cloudcover','solarradiation', 'Day sin', 'Day cos']]
  return 

def clip_power(df):
  df['power'] = df['power'].clip(lower=0, upper=2100)
  return df

def get_year_data(df):
  start_date = df.index[0]
  end_date = start_date + pd.DateOffset(months=12)
  return df[start_date:end_date]

def get_nine_mth_data(df):
  start_date = df.index[0]
  end_date = start_date + pd.DateOffset(months=9)
  return df[start_date:end_date]

def get_six_mth_data(df):
  start_date = df.index[0]
  end_date = start_date + pd.DateOffset(months=6)
  return df[start_date:end_date]

def get_three_mth_data(df):
  start_date = df.index[0]
  end_date = start_date + pd.DateOffset(months=3)
  return df[start_date:end_date]

def get_test_data(df):
  end_date = df.index[-1]
  date1 = end_date - pd.DateOffset(months=6)
  date2 = end_date - pd.DateOffset(months=3)
  return df[date1:end_date]


def create_dataset(df,window_size, forecast_size,batch_size):

    target_col = 0
    # Feel free to play with shuffle buffer size
    shuffle_buffer_size = len(df)
    # Total size of window is given by the number of steps to be considered
    # before prediction time + steps that we want to forecast
    total_size = window_size + forecast_size

    data = tf.data.Dataset.from_tensor_slices(df.values)


    # Selecting windows
    data = data.window(total_size, shift=1, drop_remainder=True)

    data = data.flat_map(lambda k: k.batch(total_size))

    # Shuffling data (seed=Answer to the Ultimate Question of Life, the Universe, and Everything)
    # data = data.shuffle(shuffle_buffer_size, seed=42)

    # Extracting past features + deterministic future + labels
    data = data.map(lambda k: (k[:-forecast_size, :], k[-forecast_size:, target_col:target_col+1])) 

    return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

def compute_error_metrics(truth_values, pred_values):

  # Calculate RMSE
  rmse = np.sqrt(mse(truth_values, pred_values))
  print("RMSE: %.3f"%rmse)

  # Calculate MAE
  vmae = mae(truth_values, pred_values)
  print("MAE: %.3f"%vmae)

  # Calculate R-squared (R2)
  rsq = r2(truth_values, pred_values)
  print("R-squared (R2): %.3f"%rsq)

  return rmse,vmae,rsq

def plot_comparision(actual, predicted):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(len(predicted))), y=predicted, mode='lines', name='Predicted value'))
    fig.add_trace(go.Scatter(x=list(range(len(actual))), y=actual, mode='lines', name='True value'))

    fig.update_layout(title='DRFN Forecast',
                      xaxis_title='Hour interval',
                      yaxis_title='Power')

    fig.show()


def plot_rec(actual, predicted):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(len(predicted))), y=predicted, mode='lines', name='Reconstruction'))
    fig.add_trace(go.Scatter(x=list(range(len(actual))), y=actual, mode='lines', name='Input'))

    fig.update_layout(title='DRFN Reconstruction',
                      xaxis_title='Hour interval',
                      yaxis_title='Power')

    fig.show()

def plot2(actual, predicted):
    fig = go.Figure()
    p2, p3 = predicted

    # Add actual line with dark color and bolder line
    fig.add_trace(go.Scatter(
        x=list(range(len(actual))),
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color='black', width=2)  # Dark color and bold line
    ))

    # Add DRFN prediction line with dashed and thinner line
    fig.add_trace(go.Scatter(
        x=list(range(len(p2))),
        y=p2,
        mode='lines',
        name='DRFN Prediction',
        line=dict(dash='dash', width=2)  # Dashed and thinner line
    ))

    # Add Persistence prediction line with dashed and thinner line
    fig.add_trace(go.Scatter(
        x=list(range(len(p2))),
        y=p3,
        mode='lines',
        name='Persistence Prediction',
        line=dict(dash='dash', width=2)  # Dashed and thinner line
    ))

    # Update layout
    fig.update_layout(
        title='Actual vs Predicted',
        xaxis_title='Hour interval',
        yaxis_title='Power'
    )

    # Show plot
    fig.show()

def plot3(actual, predicted):
    fig = go.Figure()
    p1,p2, p3 = predicted

    # Add actual line with dark color and bolder line
    fig.add_trace(go.Scatter(
        x=list(range(len(actual))),
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color='black', width=2)  # Dark color and bold line
    ))

    # Add DRFN-naive prediction line with dashed and thinner line
    fig.add_trace(go.Scatter(
        x=list(range(len(p1))),
        y=p2,
        mode='lines',
        name='Naive Prediction',
        line=dict(dash='dash', width=2)  # Dashed and thinner line
    ))

    # Add DRFN prediction line with dashed and thinner line
    fig.add_trace(go.Scatter(
        x=list(range(len(p2))),
        y=p2,
        mode='lines',
        name='DRFN Prediction',
        line=dict(dash='dash', width=2)  # Dashed and thinner line
    ))

    # Add Persistence prediction line with dashed and thinner line
    fig.add_trace(go.Scatter(
        x=list(range(len(p3))),
        y=p3,
        mode='lines',
        name='Persistence Prediction',
        line=dict(dash='dash', width=2)  # Dashed and thinner line
    ))

    # Update layout
    fig.update_layout(
        title='Actual vs Predicted',
        xaxis_title='Hour interval',
        yaxis_title='Power'
    )

    # Show plot
    fig.show()

def get_persistance_predictions(model,test,sf):
  df = test.reset_index(names='datetime')
  df['datetime'] = pd.to_datetime(df['datetime'])
  df['expected_power'] =[model.get_expected_power(x) for x in df['datetime']]
  actual = df['power'].values*sf
  predicted = df['expected_power'].fillna(0)*sf

  return actual,predicted






