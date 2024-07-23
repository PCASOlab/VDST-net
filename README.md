#VDST-NET: Disentangling spatio-temporal knowledge for weakly supervised surgical tool segmentation in video

## Description
This is an official PyTorch implementation of the VDST-Net model. 
> [VDST-Net: Disentangling spatio-temporal knowledge for weakly supervised surgical tool segmentation in video](https://arxiv.org/abs/2407.15794)

## Updates
* **2024.07.23:** Added implementation of VDST-Net, along with the data curation code of Transient Object Presence Cholec80 dataset.

## Hightlights


Weakly supervised detection/segmentation in videos could be categorized as three types. Type I has image-level presence labels available, and research for this task usually builds on WSSS adding temporal  constraints to improve prediction. Type II WSVOS with consistent object presence (COP) only has video-level labels, and the object is assumed to be in the video for most of the frames (this is the scenario for some public general video datasets). Type III WSVOS with transient object presence is the most challenging, where the label only indicates the presence of objects for the whole video, yet each object may be present in the video temporarily (i.e., in only a subset of frames).
<p align="center">
  <img src="figure/task.jpeg" width="80%" />
</p>

The core of the framework consists of a teacher-student network pair designed to disentangle spatial and temporal knowledge. Both modules utilize a Sptio-temporal CAM and share input from a VIT encoder. The primary distinction between the teacher and student lies in their upper-layer feature extraction: the teacher employs an MLP module, while the student module uses a temporal convolutional network. This design constrains the teacher's temporal interaction to overcome activation conflicts, while gradually increasing student's learning capability to reason over time.
We utilize a semi-decoupled distillation mechanism, supervising teacher and student models with identical video labels while enabling efficient knowledge transfer via gated activation maps. Despite the teacher's powerful backbone and its ability to generate activation maps with a high degree of accuracy it still has detection gaps. Although the resulting supervision signal is not perfect,  semi-decoupled distillation allows the student to use the teacher CAMs information efficiently while adding temporal context.
<p align="center">
  <img src="figure/Method.jpeg" width="70%" />
</p>

The challenge imposed by the high uncertainty of the label is amplified in the TOP situation. The results highlight the significance of semi-decouple knowledge distillation through teacher and student modules for enhanced segmentation accuracy. A standalone student module, lacking frame-level supervision, shows a notable decline in segmentation performance (Dice score from 67.80\% to 47.64\%), because it predicts error activation by taking information from wrong frames or features, pointed out by red arrowheads in the following figure. The teacher module has no such issue but lacks the ability to generate activation maps with good connectivity, as indicated by red asterisks in the figure 

<p align="center">
  <img src="figure/Ablation.jpeg" width="80%" />
</p>

Our method not only works for the surgical domain, also demonstrated superior performance on a general dataset: YouYube-Objects video data. Under the training condition of one class label per video. Our method demonstrates robustness, producing activation maps and segmentation masks that accurately follow the objects' contours, with minimal false activation on the background.  
<p align="center">
  <img src="figure/YTOBJ_Supple.jpeg" width="80%" />
</p>


## Repository Overview


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


## Comming soon
* Implementation with Resnet backbone
* Training and evaluation on the YouTube-objects dataset

 
