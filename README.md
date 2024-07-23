#VDST-NET: Disentangling spatio-temporal knowledge for weakly supervised surgical tool segmentation in video

## Description

This is an official PyTorch implementation of the VDST-Net model. 
> [VDST-NET: Disentangling spatio-temporal knowledge for weakly supervised surgical tool segmentation in video](https://arxiv.org/abs/2407.15794)
 

<p align="center">
  <img src="figure/task.jpeg" width="80%" />
</p>


<p align="center">
  <img src="figure/Method.jpeg" width="70%" />
</p>


<p align="center">
  <img src="figure/Ablation.jpeg" width="80%" />
</p>


<p align="center">
  <img src="figure/YTOBJ_Supple.jpeg" width="80%" />
</p>



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

 
