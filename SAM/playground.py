from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch
# import matplotlib.pyplot as plt
import cv2

image = cv2.imread('video01_000001_1.png')
cv2.imshow('  in put Image', image.astype((np.uint8)))
cv2.waitKey(1)
sam_checkpoint = "C:/2data/output/SAM/sam_vit_h_4b8939.pth"
sam_checkpoint = "C:/2data/output/SAM/sam_vit_l_0b3195.pth"
sam_checkpoint = "C:/2data/output/SAM/sam_vit_b_01ec64.pth"

model_type = "vit_h"
model_type = "vit_l"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)
input_point = np.array([[200, 158],[170,123]])
input_label = np.array([1,2])
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
print(masks.shape)
for i, (mask, score) in enumerate(zip(masks, scores)):
    alpha= 0.5
    # overlay = cv2.addWeighted(mask.astype((np.uint8)), 1 - alpha, image.astype((np.uint8)), alpha, 0)
    overlay = image * mask[:, :, np.newaxis]
    cv2.imshow(f"Mask {i+1}, Score: {score:.3f}", overlay.astype((np.uint8)))
    cv2.waitKey(1)
    # plt.imshow(image)
    # show_mask(mask, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    # plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    # plt.axis('off')
    # plt.show()  