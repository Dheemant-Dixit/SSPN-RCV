# Single Stage Pruned Networks (SSPNs)

## Overview

This is the code of the SSPNs introduced in the [ICCV 2023 RCV Workshop Paper](https://openaccess.thecvf.com/content/ICCV2023W/RCV/papers/Tiwari_RCV2023_Challenges_Benchmarking_Model_Training_and_Inference_for_Resource-Constrained_Deep_ICCVW_2023_paper.pdf). This project aims to classify images from the UltraMNIST dataset, which comprises large-scale images with multiple digits per image. The task is to predict the sum of the digits present in each image, ranging from 0 to 27. The project implements Single Stage Pruned Networks (SSPNs) using BatchNorm pruning of EfficientNets for efficient and accurate classification.

## Dataset

The UltraMNIST dataset consists of images that are 4000x4000 pixels in size, with 3-5 digits per image. These digits are extracted from the original MNIST dataset and combined to form complex images for classification.

## Model Architecture

The project leverages EfficientNets, a family of convolutional neural networks known for their efficiency and accuracy. BatchNorm pruning is applied to these networks to create SSPNs, which are optimized for the UltraMNIST classification task.

## Installation

To run the code, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/Dheemant-Dixit/SSPN-RCV.git
   ```
2. Move into the folder:
   ```
   cd SSPN-RCV
   ```
3. Install the required dependencies:
   ```
   conda env create -f environment.yml
   ```
4. Activate the conda environment:
   ```
   conda activate rcv_v3
   ```

## Usage

1. Download the UltraMNIST dataset in a folder names `umnist_iccv_1024` in the parent folder.
2. Train the SSPN model using the training script:
   ```
   python train.py --model_name efficientnetb2 --epochs 100
   ```
3. Finetune the SR Prune trained network:
   ```
   python finetune_pruned_network.py --model_name efficientnetb2 --epochs 100
   ```
4. Evaluate the trained model on the test set:
   ```
   python inference.py --data_dir ./umnist_iccv_1024/test/
   ```