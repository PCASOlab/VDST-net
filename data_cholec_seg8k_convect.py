import os
# import matplotlib.pyplot as plt
import cv2
# import pandas as pd
import random
import copy
import shutil
import numpy as np
Show_img =True
color_class_mapping={(127, 127, 127): 0,
                    (210, 140, 140): 1,
                    (255, 114, 114): 2,
                    (231, 70, 156): 3,
                    (186, 183, 75): 4,
                    (170, 255, 0): 5,
                    (255, 85, 0): 6,
                    (255, 0, 0): 7,
                    (255, 255, 0): 8,
                    (169, 255, 184): 9,
                    (255, 160, 165): 10,
                    (0, 50, 128): 11,
                    (111, 74, 0): 12}
class_name_mapping={0: 'Black Background',
                    1: 'Abdominal Wall',
                    2: 'Liver',
                    3: 'Gastrointestinal Tract',
                    4: 'Fat',
                    5: 'Grasper',
                    6: 'Connective Tissue',
                    7: 'Blood',
                    8: 'Cystic Duct',
                    9: 'L-hook Electrocautery',
                    10: 'Gallbladder',
                    11: 'Hepatic Vein',
                    12: 'Liver Ligament'}
# import ipywidgets as widgets
# import wandb
# Defining input images path, output path, masks path and labels path
img_size = (256, 256)  # Specify the desired size
# video_buffer_len = 29
downsample_len = 29
# Function to read labels from a text file
def read_labels(file_path):
    labels = np.genfromtxt(file_path, skip_header=1, usecols=(1, 2, 3, 4, 5, 6, 7), dtype=int)
    return labels

# Function to read frames from a video folder and resize
def read_frames(video_folder, img_size):
    frame_paths = [os.path.join(video_folder, frame) for frame in sorted(os.listdir(video_folder))]
    frames = [cv2.resize(cv2.imread(frame_path), img_size) for frame_path in frame_paths]
    return frames

path='C:/2data/cholecseg8k_working/cholecseg8k/'
op_path='C:/2data/cholecseg8k_working/'
rawimages_path=os.path.join(op_path, 'raw_images')
maskimages_path=os.path.join(op_path, 'mask_images')
labels_path=os.path.join(op_path, 'labels')
# os.makedirs(rawimages_path)
# os.makedirs(maskimages_path)
# os.makedirs(labels_path)
# Defining output train,val and test paths
imgtrainpath = os.path.join(op_path,'images','train')
imgvalpath=os.path.join(op_path,'images','validation')
imgtestpath=os.path.join(op_path,'images','test')

labeltrainpath=os.path.join(op_path,'labels','train')
labelvalpath=os.path.join(op_path,'labels','validation')
labeltestpath=os.path.join(op_path,'labels','test')

# os.makedirs(imgtrainpath)
# os.makedirs(imgvalpath)
# os.makedirs(imgtestpath)

# os.makedirs(labeltrainpath)
# os.makedirs(labelvalpath)
# os.makedirs(labeltestpath)

m=0 # variable for counting total sub-directories
n=0 # variable for counting total images
o=0 # variable for counting total raw images
p=0 # variable for counting total masks
q=0 # variable for counting total directories
video_count= 0 # for counting total videos
for directory in os.listdir(path):
    dir_path=os.path.join(path, directory)
    m=m+len(os.listdir(dir_path))
    q=q+1
    for sub_dir in os.listdir(dir_path):
        sub_dir_path=os.path.join(dir_path, sub_dir)
        n=n+len(os.listdir(sub_dir_path))
        file_list =  os.listdir(sub_dir_path)
        sorted_file_list = sorted(file_list, key=lambda x: int(x.split('_')[1]))
        video_images = []
        video_masks = []
        for image in sorted_file_list:
            src_path=os.path.join(sub_dir_path, image)
            # Rename the image based on sub-directory to distinguish images with same names from different directories
            newname=sub_dir+image
            if 'mask' not in image:   # Criterion for raw image             
                # newpath=os.path.join(rawimages_path, newname)
                # dest_path=os.path.join(rawimages_path, image)
                # shutil.copy(src_path, dest_path) # Copying raw image to output directory
                # os.rename(dest_path, newpath) # Renaming raw image
                o=o+1
                this_image=cv2.resize(cv2.imread(src_path), img_size)
                video_images.append(this_image)
            if 'color_mask' in image:  # Criterion for color mask  
                # newpath=os.path.join(maskimages_path, newname)
                # dest_path=os.path.join(maskimages_path, image)
                # shutil.copy(src_path, dest_path) # Copying mask to output directory
                # os.rename(dest_path, newpath) # Renaming mask
                p=p+1
                this_mask=cv2.resize(cv2.imread(src_path), img_size,cv2.INTER_AREA)
                this_mask = cv2.cvtColor(this_mask, cv2.COLOR_BGR2RGB)

                mask_12_channel = np.zeros((len(class_name_mapping),*img_size), dtype=np.uint8)  # Initialize 12-channel mask

                # Create a mask for each color class
                for idx, color in enumerate(color_class_mapping.keys()):
                    class_mask = np.all(this_mask == color, axis=-1)
                    mask_12_channel[idx, :, :] = class_mask.astype(np.uint8) 
                    # if Show_img:
                    #     cv2.imshow(class_name_mapping[idx],class_mask.astype(np.uint8) *255)
                    #     cv2.waitKey(1)

                # this_mask = cv2.cvtColor(this_mask, cv2.COLOR_BGR2RGB)

                video_masks.append(mask_12_channel)
                if Show_img == True:
                    cv2.imshow('color maks',this_mask.astype(np.uint8) *255)

                    cv2.imshow('Grasp',mask_12_channel[5].astype(np.uint8) *255)
                    cv2.imshow('hook',mask_12_channel[9].astype(np.uint8) *255)

                    cv2.waitKey(1)

        print(len(video_images))    
        print(len(video_masks))   
        video_images = np.array(video_images)
        video_masks = np.array(video_masks)

        # Ensure the downsampled video has exact length of downsample_len
        if len(video_images) > downsample_len:
            step = len(video_images) // downsample_len
            video_images = video_images[:downsample_len * step:step]
        if len(video_masks) > downsample_len:
            step = len(video_masks) // downsample_len
            video_masks = video_masks[:downsample_len * step:step]
        video_images  = np.transpose(video_images , (3, 0, 1, 2))  # Reshape to (3, 29, 64, 64)
        print(video_images.shape)



        print(video_masks.shape) 



print("Total number of sub-directories:", m)
print("Total number of images:", n)
print("Total number of raw images:", o)
print("Total number of masks:", p)

len(os.listdir(rawimages_path)), len(os.listdir(maskimages_path))



# for directory in os.listdir(path):
#     dir_path=os.path.join(path, directory)
#     m=m+len(os.listdir(dir_path))
#     q=q+1
#     for sub_dir in os.listdir(dir_path):
#         sub_dir_path=os.path.join(dir_path, sub_dir)
#         n=n+len(os.listdir(sub_dir_path))
#         file_list =  os.listdir(sub_dir_path)
#         sorted_file_list = sorted(file_list, key=lambda x: int(x.split('_')[1]))
#         for image in sorted_file_list:
#             src_path=os.path.join(sub_dir_path, image)
#             # Rename the image based on sub-directory to distinguish images with same names from different directories
#             newname=sub_dir+image
#             if 'mask' not in image:   # Criterion for raw image             
#                 newpath=os.path.join(rawimages_path, newname)
#                 dest_path=os.path.join(rawimages_path, image)
#                 shutil.copy(src_path, dest_path) # Copying raw image to output directory
#                 os.rename(dest_path, newpath) # Renaming raw image
#                 o=o+1
#             if 'color_mask' in image:  # Criterion for color mask  
#                 newpath=os.path.join(maskimages_path, newname)
#                 dest_path=os.path.join(maskimages_path, image)
#                 shutil.copy(src_path, dest_path) # Copying mask to output directory
#                 os.rename(dest_path, newpath) # Renaming mask
#                 p=p+1