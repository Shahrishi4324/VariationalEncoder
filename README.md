# Variational Autoencoder with PyTorch

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)

## Overview

This project demonstrates the implementation of a Variational Autoencoder (VAE) using PyTorch. The VAE is trained on the MNIST dataset, which contains images of handwritten digits. The model learns to encode the images into a latent space and decode them back into images, with the ability to generate new digit images by sampling from the latent space.

## Dataset

The project uses the MNIST dataset, a well-known dataset in the field of computer vision. The dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0-9).

## Features

- **Encoder-Decoder Architecture:** Implementing the VAE with an encoder to learn the latent space and a decoder to reconstruct the images.
- **Training with Reconstruction Loss and KL Divergence:** Training the VAE using a combination of reconstruction loss and KL divergence to regularize the latent space.
- **Sampling from the Latent Space:** Generating new images by sampling from the learned latent space.

## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/RishiShah99/vae_pytorch.git
   ```
2. Run the program
