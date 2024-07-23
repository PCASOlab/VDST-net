
# the model
import cv2
import numpy

import os
import shutil
# from train_display import *
# the model
# import arg_parse
import cv2
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from model import  model_experiement, model_infer
from working_dir_root import Output_root,Save_flag,Load_flow
from dataset.dataset import myDataloader,categories
from dataset import io
# Save_flag =False
def save_img_to_folder(this_save_dir,ID,img):
    # this_save_dir = Output_root + "1out_img/" + Model_key + "/ground_circ/"
    if not os.path.exists(this_save_dir):
        os.makedirs(this_save_dir)
    cv2.imwrite(this_save_dir +
                str(ID) + ".jpg", img)

class Display(object):
    def __init__(self,GPU=False):
        self.Model_infer = model_infer._Model_infer(GPU)
        self.dataLoader = myDataloader()
        self.show_num=15

    def train_display(self,MODEL_infer,mydata_loader, read_id):
        # copy all the input videos and labels
        # cv2.destroyAllWindows()
        self.Model_infer.output= MODEL_infer.output
        # self.Model_infer.slice_valid = MODEL_infer.slice_valid
        self.Model_infer.cam3D = MODEL_infer.cam3D
        self.dataLoader.input_videos = mydata_loader.input_videos
        self.dataLoader.labels = mydata_loader.labels
        self.dataLoader.input_flows = mydata_loader.input_flows
        self.Model_infer.input_resample = MODEL_infer.input_resample

        if Load_flow == True:
            Gray_video = self.dataLoader.input_flows[0,:,:,:] # RGB together
            Ori_D,Ori_H,Ori_W = Gray_video.shape
            step_l = int(Ori_D/self.show_num)+1
            for i in range(0,Ori_D,step_l):
                if i ==0:
                    stack = Gray_video[i]
                else:
                    stack = np.hstack((stack,Gray_video[i]))

            # Display the final image
            cv2.imshow('Stitched in put flows', stack.astype((np.uint8)))
            cv2.waitKey(1)


        # Gray_video = self.Model_infer.input_resample[0,2,:,:,:].cpu().detach().numpy()# RGB together
        Gray_video = self.dataLoader.input_videos[0,0,:,:,:] # RGB together
        Ori_D,Ori_H,Ori_W = Gray_video.shape
        step_l = int(Ori_D/self.show_num)+1
        for i in range(0,Ori_D,step_l):
            if i ==0:
                stack1 =  Gray_video[i] 
            else:
                stack1 = np.hstack((stack1,Gray_video[i]))

        # Display the final image
        # cv2.imshow('Stitched in put Image', stack1.astype((np.uint8)))
        # cv2.waitKey(1)

        if Save_flag == True:
            io.save_img_to_folder(Output_root + "image/original/" ,  read_id, stack1.astype((np.uint8)) )
        # Combine the rows vertically to create the final 3x3 arrangement
        Cam3D=self.Model_infer.cam3D[0]
        label_0 = self.dataLoader.labels[0]
        if len (Cam3D.shape) == 3:
            Cam3D = Cam3D.unsqueeze(1)
        ch, D, H, W = Cam3D.size()
        # activation = nn.Sigmoid()
        # Cam3D =  activation( Cam3D)
        # average_tensor = Cam3D.mean(dim=[1,2,3], keepdim=True)
        # _, sorted_indices = average_tensor.sort(dim=0)
        if len (self.Model_infer.output.shape) == 5:
            output_0 = self.Model_infer.output[0,:,0,0,0].cpu().detach().numpy()
        else:
            output_0 = self.Model_infer.output[0,:,0,0].cpu().detach().numpy()
        step_l = int(D/self.show_num)+1
        stitch_i =0
        stitch_im  = np.zeros((H,W))
        stitch_over = np.zeros((H,W))
        for j in range(len(categories)):
            # j=sorted_indices[13-index,0,0,0].cpu().detach().numpy()
            this_grayVideo = Cam3D[j].cpu().detach().numpy()
            if (output_0[j]>0.5 or label_0[j]>0.5):
                for i in range(0, D, step_l):
                    this_image = this_grayVideo[i]
                    this_image =  cv2.resize(this_image, (Ori_H, Ori_W), interpolation = cv2.INTER_LINEAR)
            
                    if i == 0:
                        stack = this_image
                    else:
                        stack = np.hstack((stack, this_image))
                stack = stack -np.min(stack)
                stack = stack /(np.max(stack)+0.0000001)*254
                # stack = (stack>20)*stack
                # stack = (stack>0.5)*128
                stack = np.clip(stack,0,254)
                alpha= 0.5
                overlay = cv2.addWeighted(stack1.astype((np.uint8)), 1 - alpha, stack.astype((np.uint8)), alpha, 0)
                # stack =  stack - np.min(stack)
                infor_image = this_image*0
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                font_color = (255, 255, 255)  # White color

                text1 = str(j) + "S"+ "{:.2f}".format(output_0[j])  
                
                text2="G"+ str(label_0[j])
                text3 = categories[j]
                # Define the position where you want to put the text (bottom-left corner)
                text_position = (5, 20)
                # Use cv2.putText() to write the text on the image
                cv2.putText(infor_image, text1, text_position, font, font_scale, font_color, font_thickness)
                text_position = (5, 30)
                # Use cv2.putText() to write the text on the image
                cv2.putText(infor_image, text2, text_position, font, font_scale, font_color, font_thickness)
                text_position = (5, 40)
                # Use cv2.putText() to write the text on the image
                cv2.putText(infor_image, text3, text_position, font, font_scale, font_color, font_thickness)
                # stack = stack -np.min(stack)
                # stack = stack /(np.max(stack)+0.0000001)*254
                stack = np.hstack((infor_image, stack))
                overlay = np.hstack((infor_image, overlay))
               
                # Display the final image
                # cv2.imshow( str(j) + "score"+ "{:.2f}".format(output_0[j]) + "GT"+ str(label_0[j])+categories[j], stack.astype((np.uint8)))
                # cv2.waitKey(1)
                if stitch_i ==0:
                    stitch_im = stack
                    stitch_over = overlay
                else:
                    stitch_im = np.vstack((stitch_im, stack))
                    stitch_over = np.vstack((stitch_over, overlay))

                stitch_i+=1

        image_all = np.vstack((stitch_over,stitch_im))
        cv2.imshow( 'all', image_all.astype((np.uint8)))
        # cv2.imshow( 'overlay', stitch_over.astype((np.uint8)))

        cv2.waitKey(1)
        if Save_flag == True:

            io.save_img_to_folder(Output_root + "image/predict/" ,  read_id, stitch_im.astype((np.uint8)) )
            io.save_img_to_folder(Output_root + "image/predict_overlay/" ,  read_id, image_all.astype((np.uint8)) )


        if MODEL_infer.gradcam is not None:
            heatmap = MODEL_infer.gradcam[0,0,:,:].cpu().detach().numpy()

            # heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-5)

                # Resize the heatmap to the original image size
            # heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))

            # Apply colormap to the heatmap
            heatmap_colormap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

            # Superimpose the heatmap on the original image
            # result = cv2.addWeighted(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 0.7, heatmap_colormap, 0.3, 0)

            # Display the result
            cv2.imshow('ST-CAM', heatmap_colormap)
            cv2.waitKey(1)
        # Cam3D = nn.functional.interpolate(side_out_low, size=(1, Path_length), mode='bilinear')


