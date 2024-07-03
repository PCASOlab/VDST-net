# Code for "Disentangling spatio-temporal information for weakly supervised surgical tool segmentation in video"

## Description

This repository contains scripts for implemention of our paper's training pipeline for video-level weakly supervised segmentation. The key scripts included are:

 

## Setup and Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- PyTorch
- torchvision
- PIL
- OpenCV

### Installation

1. Clone this repository 
 
 Ensure you have the necessary directories and data in place:
 
Usage
1. Dataset Script: data_cholec_reader_convect.py
Convert the raw cholec80 data into pkls of videos clips.

download data:
Cholec80 dataset: [Download Cholec80](https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.tar.gz)
 

2. Training Script: main.py
This script trains the  model on the specified dataset. 
 

To train the model, 
first set your data storage dir in working_dir_roor.py 
run:

python main.py --evaluation False


To evaluate the model, run:

python main.py --evaluation True

 
