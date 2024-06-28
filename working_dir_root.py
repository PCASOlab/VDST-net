import argparse

# Define the argument parser
parser = argparse.ArgumentParser(description='Script for configuring dataset paths and parameters.')

# Adding arguments
parser.add_argument('--working_root', type=str, default="C:/2data/", help='Root working directory')
parser.add_argument('--evaluation', type=bool, default=True, help='Evaluation mode')
parser.add_argument('--img_size', type=int, default=64, help='Image size')
parser.add_argument('--gpu_mode', type=bool, default=True, help='GPU mode')
parser.add_argument('--continue_flag', type=bool, default=True, help='Continue flag')
parser.add_argument('--visdom_flag', type=bool, default=True, help='Visdom flag')
parser.add_argument('--display_flag', type=bool, default=True, help='Display flag')
parser.add_argument('--save_flag', type=bool, default=True, help='Save flag')
parser.add_argument('--loadmodel_index', type=str, default='2.pth', help='Load model index')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--data_aug', type=bool, default=False, help='Data augmentation')
parser.add_argument('--random_mask', type=bool, default=False, help='Random mask')
parser.add_argument('--random_full_mask', type=bool, default=False, help='Random full mask')
parser.add_argument('--load_feature', type=bool, default=True, help='Load feature')
parser.add_argument('--save_feature_olg', type=bool, default=True, help='Save feature OLG')
parser.add_argument('--load_flow', type=bool, default=False, help='Load flow')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay')
parser.add_argument('--max_lr', type=float, default=0.001, help='Max learning rate')
parser.add_argument('--learningr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--learningr_res', type=float, default=0.00001, help='Learning rate for residuals')
parser.add_argument('--call_gradcam', type=bool, default=False, help='Call GradCAM')

# Parse arguments
args = parser.parse_args()

# Assigning the arguments to the variables
working_root = args.working_root

Dataset_video_root = working_root + "training_data/video_clips/"
Dataset_video_pkl_root = working_root + "training_data/video_clips_pkl/"
Dataset_video_pkl_flow_root = working_root + "training_data/video_clips_pkl_flow/"
Dataset_video_pkl_cholec = working_root + "training_data/video_clips_pkl_cholec/"
Dataset_label_root = working_root + "training_data/"
config_root = working_root + "config/"
Output_root = working_root + "output/"
SAM_pretrain_root = working_root + "output/SAM/"
output_folder_sam_feature = working_root + "cholec80/output_sam_features/"

train_test_list_dir = working_root + "output/train_test_list/"
train_sam_feature_dir = working_root + "cholec80/train_sam_feature/"
sam_feature_olg_dir = working_root + "cholec80/sam_feature_OLG/"

Evaluation = args.evaluation
img_size = args.img_size
GPU_mode = args.gpu_mode
Continue_flag = args.continue_flag if args.evaluation else True
Visdom_flag = args.visdom_flag
Display_flag = args.display_flag
Save_flag = args.save_flag
loadmodel_index = args.loadmodel_index

Batch_size = args.batch_size
Data_aug = args.data_aug
Random_mask = args.random_mask
Random_Full_mask = args.random_full_mask
Load_feature = args.load_feature
Save_feature_OLG = args.save_feature_olg

if Load_feature:
    Save_feature_OLG = False
if Save_feature_OLG:
    Batch_size = 1

Load_flow = args.load_flow
Weight_decay = args.weight_decay
Max_lr = args.max_lr
learningR = args.learningr
learningR_res = args.learningr_res
Call_gradcam = args.call_gradcam

class Para(object):
    def __init__(self):
        self.x = 0

# Example usage: python script.py --working_root "path/to/root" --evaluation False
