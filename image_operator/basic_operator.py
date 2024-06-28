import cv2
import numpy as np
import os
from random import random, choice
# import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
def random_verse_the_video(video,max=255):
    flag = choice([True, False])
    if flag == True:
        output_v = max - video 
    else:
        output_v = video
    return output_v


def random_augment(video):
   
    ch, D, H, W= video.shape
    alpha = random() * 0.8 + 0.2  # random between 0.2 and 0.5
    angle = choice([90,180,270 ])  # random choose from 0, 90, -90

    # Reshape the stack of images to (180, 256, 256)
    images = video.reshape(ch * D, H, W)

    # Function to apply data augmentation
    def apply_data_augmentation(image):
        # Convert to PIL Image
        pil_image = Image.fromarray(np.uint8(image ))

        # Data augmentation: Random Saturation and Random Rotation
        data_augmentation = transforms.Compose([
            transforms.ColorJitter(saturation=alpha),
             
        ])
        augmented_image= transforms.functional.rotate (pil_image, angle)
        # Apply data augmentation
        augmented_image = data_augmentation(augmented_image)
         
        # Convert back to NumPy array
        augmented_image = np.array(augmented_image) 

        return augmented_image

    # Apply data augmentation to each image in the stack
    augmented_images = np.stack([apply_data_augmentation(image) for image in images])

    # Reshape augmented_images back to (3, 60, 256, 256)
    augmented_images = augmented_images.reshape(ch, D, H, W)

    return augmented_images,angle

def rotate_buff(images,angle):
    D, H, W= images.shape
  

    # Function to apply data augmentation
    def apply_data_augmentation(image):
        # Convert to PIL Image
        pil_image = Image.fromarray(np.uint8(image ))

        # Data augmentation: Random Saturation and Random Rotation
         
        augmented_image= transforms.functional.rotate (pil_image, angle)
        # Apply data augmentation
      
         
        # Convert back to NumPy array
        augmented_image = np.array(augmented_image) 

        return augmented_image

    # Apply data augmentation to each image in the stack
    augmented_images = np.stack([apply_data_augmentation(image) for image in images])

    # Reshape augmented_images back to (3, 60, 256, 256)
     

    return augmented_images 


def hide_patch(video, patch_num=32, hide_prob=0.5, mean=124,image_level= True):
    # assume patch_num is int**2
    if patch_num == 1: return video
    flag = choice([True, False])
    
    if flag == True:
       return video
    ch, D, H, W = video.shape
    pn = int(patch_num ** (1/2))
    patch_size = int(W // pn)
    patch_offsets = [(x * patch_size, y * patch_size) for x in range(pn) for y in range(pn)]

    # if np.random.uniform() < hide_prob:
    #     for d in range(D):
    #         for (px, py) in patch_offsets:
    #             video[:, d, px:px + patch_size, py:py + patch_size] = mean
    
    if image_level == False:  
        for (px, py) in patch_offsets:
            if np.random.uniform() < hide_prob:
                    video[:, :, px:px + patch_size, py:py + patch_size] = mean
    else:
      for i in range(D):
        for (px, py) in patch_offsets:
            if np.random.uniform() < hide_prob:
                    video[:, i, px:px + patch_size, py:py + patch_size] = mean
    return video


def hide_full_image(video,  hide_prob=0.5, mean=124 ):
    # assume patch_num is int**2
   
    flag = choice([True, False])
    
    if flag == True:
       return video
    ch, D, H, W = video.shape
    
    for i in range(D):
        if np.random.uniform() < hide_prob:
                video[:, i, :, :] = mean
    return video

def motion_map(video):
    ch, D, H, W = video.shape
    shifted = np.roll(video, 1, axis=1)
    motion = np.abs(video - shifted)
    motion = (motion - np.min(motion)) / np.max(motion) * 255
    return motion


def compute_optical_flow(video):
    ch, D, H, W = video.shape
    motion_map = np.zeros((D, H, W), dtype=np.float32)

    prev_frame = video[0, 0, :, :]  # Use the first frame as the initial frame

    for i in range(1, D):
        current_frame = video[0, i, :, :]
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, current_frame, flow=None, pyr_scale=0.5, levels=5, winsize=7, iterations=10, poly_n=5, poly_sigma=1.1, flags=0
        )

        # Calculate magnitude of the flow vectors
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # Normalize and scale the magnitude to the range [0, 255]
        magnitude = (magnitude - np.min(magnitude)) / np.max(magnitude) * 255

        # Store the computed motion map
        motion_map[i, :, :] = magnitude

        # Update the previous frame
        prev_frame = current_frame
    motion_map[0, :, :] = motion_map[1, :, :] 
    return motion_map