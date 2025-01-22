## DRFN - Deep Recurrent Fusion Network

This repository contains the code for the Deep Reconstruction Forecast Network (DRFN) architecture presented in our paper titled "Unsupervised Domain Adaptation Framework for Photovoltaic Power Forecasting Using Variational Auto-Encoders".The code has been made available to support reproducibility and foster further research in renewable energy (RE) forecasting.
The DRFN framework addresses critical challenges in day-ahead energy production forecasting, particularly for newly installed photovoltaic (PV) plants where labeled historical data is unavailable. By employing a feature-based unsupervised domain adaptation approach, the DRFN architecture enables effective knowledge transfer without relying on adversarial training, offering a practical and computationally efficient alternative.

Our approach provides significant contributions, including:
- Introducing a novel DRFN architecture that extends the Deep Reconstruction Classification Network (DRCN).
- Demonstrating unsupervised domain adaptation for forecasting without labeled target domain data.

This repository includes all the tools required to train and test the DRFN model, along with utilities for data preprocessing and customization.

## Deep Reconstruction Forecast Network (DRFN) 

The proposed Deep Reconstruction Forecast Network is designed to perform time-series forecasting with a focus on extracting discriminative and robust features. It integrates a Variational Autoencoder (VAE) architecture for stochastic feature extraction. The network is composed of three main components:

- Encoder: Extracts features from the input data and maps them to a latent feature space. This space is modeled as a Gaussian distribution, with the encoder estimating its mean and standard deviation.
- Decoder: Reconstructs the input data from the latent feature space, ensuring that the features retain the intrinsic structure of the input data.
- Forecaster: Maps the features in the latent space to the forecast labels, enabling accurate prediction.

![Model Architecture](DRFN.jpeg)

## Transfer Learning with \acrshort{dfrn}

The \acrshort{dfrn} enables domain adaptation through a two-step process:

### Training in the Source Domain
- The encoder, decoder, and forecaster are trained together using labeled data from the source domain.
- This step establishes a robust latent feature space and a forecasting model optimized for the source data.

### Adaptation to the Target Domain
1. In the target domain, where labeled data may be limited or unavailable, the forecaster trained on the source domain remains unchanged.
2. The encoder-decoder pair is fine-tuned to align the feature space of the target domain with that of the source domain. 
   - This alignment is achieved by minimizing the divergence between the feature distributions of the two domains.
3. The alignment ensures that the forecaster can seamlessly operate on the target domain without requiring additional supervised training.

![Model Architecture](DRFN_TL.jpeg)


## Setup and Installation

To use this code, you will need to set up a Python virtual environment and install the necessary dependencies. Please follow the steps below:

### Create and activate Virtual Environment

Create a new Python virtual environment. You can do this by running the following commands:

```bash
# Create the virtual environment (change 'env_name' to your preferred name)
python3 -m venv env_name


```bash
# Create the virtual environment (change 'env_name' to your preferred name)
source env_name/bin/activate
