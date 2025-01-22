
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers

class DRFN_source(keras.Model):
    def __init__(self, encoder, decoder, forecaster, **kwargs):
        super(DRFN_source,self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.forecaster = forecaster
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.forecasting_loss_tracker = keras.metrics.Mean(
            name="forecast_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.forecasting_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            forecast = self.forecaster(z)
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.mean_squared_error(data[0][..., :1], reconstruction),
                    axis=(0, 1),
                )
            )
            forecast_loss = ops.mean(
                ops.sum(
                    keras.losses.mean_squared_error(data[1], forecast),
                    axis=(0, 1),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = forecast_loss + reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.forecasting_loss_tracker.update_state(forecast_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "forecast_loss": self.forecasting_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def call(self, inputs,training=False):
        """
        Define the forward pass for inference.
        Args:
            inputs: Input data, assumed to be the past time series.
        
        Returns:
            reconstruction: Reconstructed input.
            forecast: Predicted future values.
        """
        z_mean, z_log_var, z = self.encoder(inputs[0],training=False)
        reconstruction = self.decoder(z,training=False)
        forecast = self.forecaster(z,training=False)
        return reconstruction, forecast
    
    def get_config(self):
        # Return a dictionary containing all arguments required to reconstruct the model
        config = super().get_config()
        config.update({
            "encoder": keras.layers.serialize(self.encoder),
            "decoder": keras.layers.serialize(self.decoder),
            "forecaster": keras.layers.serialize(self.forecaster),
        })
        return config
    
    def build(self, input_shape, layer_init):
        # Note the customization in overriding `build()` adds an extra argument.
        # Therefore, we will need to manually call build with `layer_init` argument
        # before the first execution of `call()`.
        super().build(input_shape)
        self.layer_init = layer_init

    @classmethod
    def from_config(cls, config):
        # Reconstruct the layers from their serialized configurations
        encoder = keras.layers.deserialize(config.pop("encoder"))
        decoder = keras.layers.deserialize(config.pop("decoder"))
        forecaster = keras.layers.deserialize(config.pop("forecaster"))
        return cls(encoder=encoder, decoder=decoder, forecaster=forecaster, **config)


class DRFN_target(keras.Model):
    def __init__(self, encoder_source, encoder, decoder, forecaster, **kwargs):
        super(DRFN_target,self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.forecaster = forecaster
        self.encoder_source = encoder_source
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.forecasting_loss_tracker = keras.metrics.Mean(
            name="forecast_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.forecasting_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data[0])
            s_z_mean , s_log_var , s_z = self.encoder_source(data[0],training=False)
            reconstruction = self.decoder(z)
            forecast = self.forecaster(z,training=False)
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.mean_squared_error(data[0][..., :1], reconstruction),
                    axis=(0, 1),
                )
            )
            forecast_loss = ops.mean(
                ops.sum(
                    keras.losses.mean_squared_error(data[1], forecast),
                    axis=(0, 1),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            # kl_loss = -0.5 * ((ops.square(s_z_mean-z_mean)/z_log_var)
            #                     +(s_log_var/z_log_var)
            #                     -1
            #                     -ops.exp(s_log_var)
            #                     +ops.exp(z_log_var))
            
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = (1-0.7)*reconstruction_loss + 0.7*kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.forecasting_loss_tracker.update_state(forecast_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "forecast_loss": self.forecasting_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def call(self, inputs,training=False):
        """
        Define the forward pass for inference.
        Args:
            inputs: Input data, assumed to be the past time series.
        
        Returns:
            reconstruction: Reconstructed input.
            forecast: Predicted future values.
        """
        z_mean, z_log_var, z = self.encoder(inputs[0],training=False)
        reconstruction = self.decoder(z,training=False)
        forecast = self.forecaster(z,training=False)
        return reconstruction, forecast
    
