import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.model_3dcnn_linear_TC import _VideoCNN
from model.model_3dcnn_linear_ST import _VideoCNN_S
from working_dir_root import learningR,learningR_res,SAM_pretrain_root,Load_feature,Weight_decay,Evaluation
from dataset.dataset import class_weights
from SAM.segment_anything import  SamPredictor, sam_model_registry
# from MobileSAM.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from dataset.dataset import label_mask,Mask_out_partial_label
if Evaluation == True:
    learningR=0
    Weight_decay=0
# learningR = 0.0001
class _Model_infer(object):
    def __init__(self, GPU_mode =True,num_gpus=1):
        if GPU_mode ==True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            device = torch.device("cpu")
        self.device = device
        sam_checkpoint = SAM_pretrain_root+"sam_vit_h_4b8939.pth"
        sam_checkpoint = SAM_pretrain_root+"sam_vit_l_0b3195.pth"
        sam_checkpoint =SAM_pretrain_root+ "sam_vit_b_01ec64.pth"
        self.inter_bz =2
        model_type = "vit_h"
        model_type = "vit_l"
        model_type = "vit_b"
        
        # model_type = "vit_t"
        # sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # self.predictor = SamPredictor(self.sam) 
        self.Vit_encoder = sam.image_encoder
        sam_predictor = SamPredictor(sam)
        self.sam_model = sam_predictor.model
        self.VideoNets = _VideoCNN()
        self.VideoNets_S = _VideoCNN_S()
        self.input_size = 1024
        resnet18 = models.resnet18(pretrained=True)
        self.gradcam = None
        
        # Remove the fully connected layers at the end
        partial = nn.Sequential(*list(resnet18.children())[0:-2])
        
        # Modify the last layer to produce the desired feature map size
        self.resnet = nn.Sequential(
            partial,
            nn.ReLU()
        )
        # if GPU_mode ==True:
        #     self.VideoNets.cuda()
        
        if GPU_mode == True:
            if num_gpus > 1:
                # self.VideoNets.classifier = torch.nn.DataParallel(self.VideoNets.classifier)
                # self.VideoNets.blocks = torch.nn.DataParallel(self.VideoNets.blocks)
                self.VideoNets = torch.nn.DataParallel(self.VideoNets)
                self.VideoNets_S = torch.nn.DataParallel(self.VideoNets_S)


                self.resnet  = torch.nn.DataParallel(self.resnet )
                self.Vit_encoder   = torch.nn.DataParallel(self.Vit_encoder  )
                self.sam_model  = torch.nn.DataParallel(self.sam_model )
        self.VideoNets.to(device)
        self.VideoNets_S.to(device)


        # self.VideoNets.classifier.to(device)
        # self.VideoNets.blocks.to(device)


        self.resnet .to(device)
        self.Vit_encoder.to(device)
        self.sam_model .to (device)
        if Evaluation:
             self.VideoNets.eval()
             self.VideoNets_S.eval()

        
        weight_tensor = torch.tensor(class_weights, dtype=torch.float)
        # self.customeBCE = torch.nn.BCEWithLogitsLoss().to(device)
        self.customeBCE = torch.nn.BCEWithLogitsLoss(weight=weight_tensor).to(device)
        self.customeBCE_mask = torch.nn.MSELoss( ).to(device)

        # self.customeBCE = torch.nn.BCELoss(weight=weight_tensor).to(device)
        
        # self.optimizer = torch.optim.Adam([
        # {'params': self.VideoNets.parameters(),'lr': learningR}
        # # {'params': self.VideoNets.blocks.parameters(),'lr': learningR*0.9}
        # ], weight_decay=0.1)
        # self.optimizer = torch.optim.Adam([
        # {'params': self.VideoNets.parameters(),'lr': learningR}
        # # {'params': self.VideoNets.blocks.parameters(),'lr': learningR*0.9}
        # ])
        self.optimizer = torch.optim.AdamW ([
        {'params': self.VideoNets.parameters(),'lr': learningR}
        # {'params': self.VideoNets.blocks.parameters(),'lr': learningR*0.9}
        ], weight_decay=Weight_decay)
        self.optimizer_s = torch.optim.AdamW ([
        {'params': self.VideoNets_S.parameters(),'lr': learningR}
        # {'params': self.VideoNets.blocks.parameters(),'lr': learningR*0.9}
        ], weight_decay=Weight_decay)
        # if GPU_mode ==True:
        #     if num_gpus > 1:
        #         self.optimizer = torch.nn.DataParallel(optself.optimizerimizer)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def forward(self,input,input_flows, features):
        # self.res_f = self.resnet(input)
        bz, ch, D, H, W = input.size()

        self.input_resample =   F.interpolate(input,  size=(D, self.input_size, self.input_size), mode='trilinear', align_corners=False)
        # self.
        if Load_feature == False:
            flattened_tensor = self.input_resample.permute(0,2,1,3,4)
            flattened_tensor = flattened_tensor.reshape(bz * D, ch, self.input_size, self.input_size)
            flattened_tensor = (flattened_tensor-124.0)/60.0

            num_chunks = (bz*D + self.inter_bz - 1) // self.inter_bz
        
            # List to store predicted tensors
            predicted_tensors = []
            
            # Chunk input tensor and predict
            with torch.no_grad():
                for i in range(num_chunks):
                    start_idx = i * self.inter_bz
                    end_idx = min((i + 1) * self.inter_bz, bz*D)
                    input_chunk = flattened_tensor[start_idx:end_idx]
                    output_chunk = self.Vit_encoder(input_chunk)
                    predicted_tensors.append(output_chunk)
                    # torch.cuda.empty_cache()
               
        
            # Concatenate predicted tensors along batch dimension
            concatenated_tensor = torch.cat(predicted_tensors, dim=0)

            flattened_tensor = concatenated_tensor
            new_bz, new_ch, new_H, new_W = flattened_tensor.size()
            self.f = flattened_tensor.reshape (bz,D,new_ch,new_H, new_W).permute(0,2,1,3,4)
        else:
            with torch.no_grad():
                self.f = features
        self.output, self.slice_valid, self. cam3D= self.VideoNets(self.f,input_flows)
        with torch.no_grad():
            self.slice_hard_label,self.binary_masks= self.CAM_to_slice_hardlabel(self.cam3D)
            self.cam3D_target = self.cam3D.detach().clone()
        self.output_s,self.slice_valid_s,self.cam3D_s = self.VideoNets_S(self.f,input_flows)
        # self.sam_mask_prompt_decode(self.cam3D,self.f)
        # self.cam3D = self. cam3D_s
    def CAM_to_slice_hardlabel(self,cam):
        bz, ch, D, H, W = cam.size()
        raw_masks = cam -torch.min(cam)
        raw_masks = raw_masks /(torch.max(raw_masks)+0.0000001)        
        binary_mask = (raw_masks >0.05)*1.0
        binary_mask = self. clear_boundary(binary_mask)
        # flatten_mask = binary_mask.view(bz,ch)
        count_masks = torch.sum(binary_mask, dim=(-1, -2), keepdim=True)
        slice_hard_label = (count_masks>10)*1.0
        return slice_hard_label,binary_mask

    def sam_mask_prompt_decode(self,raw_masks,features ,multimask_output: bool = False):
        bz, ch, D, H, W = raw_masks.size()
        bz_f, ch_f, D_f, H_f, W_f = features.size()

        raw_masks = raw_masks -torch.min(raw_masks)
        raw_masks = raw_masks /(torch.max(raw_masks)+0.0000001) 
        self.mask_resample =   F.interpolate(raw_masks,  size=(D, 256, 256), mode='trilinear', align_corners=False)
        binary_mask = (self.mask_resample >0.1)*1.0
        # binary_mask =  self.mask_resample 

        # binary_mask = binary_mask.float(). to (self.device)
        # flattened_tensor = binary_mask.reshape(bz *ch* D,  256, 256)
        flattened_feature = features.permute(0,2,1,3,4)
        flattened_feature = flattened_feature.reshape(bz_f * D_f, ch_f, H_f, W_f)

        flattened_mask= binary_mask.permute(0,2,1,3,4)
        flattened_mask = flattened_mask.reshape(bz * D, ch, 256, 256)
        output_mask = flattened_mask*0

        with torch.no_grad():
                for i in range(ch):
                    for j in range (bz*D):
                        this_input_mask =  flattened_mask[j,i,:,:]
                        this_feature= flattened_feature[j:j+1,:,:,:]
                        coordinates = torch.ones(bz * D,1,2)*512.0
                        coordinates= coordinates.cuda()
                        labels = torch.ones(bz * D,1)
                        forground_num =  int(torch.sum(this_input_mask).item())
                        if forground_num>41:
                            foreground_indices = torch.nonzero(this_input_mask > 0.5, as_tuple=False)
                            cntral = self.extract_central_point_coordinates(this_input_mask)
                                # Extract coordinates from indices
                            foreground_coordinates = foreground_indices[:, [1, 0]]  # Swap x, y to get (y, x) format
                            labels = torch.ones(1,1)
                            coordinates = cntral.view(1,1,2)*4
                            coordinates= coordinates.cuda()
                            labels = labels.cuda()


                        # sampled_coordinates, sampled_labels = self.sample_points(this_input_mask, num_points=16)
                        # sampled_coordinates= sampled_coordinates.cuda() *4
                    # flat_mask =  this_input_mask.view(bz*D, -1)

# Find the index of the maximum value
                    # max_indices = torch.argmax(flat_mask, dim=1)

                    # # Convert indices to coordinates
                    # y_coordinates = max_indices // W
                    # x_coordinates = max_indices % W
                    # coordinates = torch.stack((y_coordinates, x_coordinates), dim=1).unsqueeze(1)*4

                    # coordinates = max_indices.unsqueeze(-1)
                            points = (coordinates, labels)

                            # Embed prompts
                            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                                points=points,
                                boxes=None,
                                masks=None,
                            )
                            # Predict masks
                            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                                image_embeddings= this_feature,
                                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=multimask_output,
                            )
                            output_mask[j,i,:,:] = low_res_masks[:,0,:,:]>0


        # self.f = flattened_tensor.reshape (bz,D,new_ch,new_H, new_W).permute(0,2,1,3,4)
        self.sam_mask = output_mask.reshape (bz,D,ch,256,256).permute(0,2,1,3,4)
        # self.sam_mask = binary_mask

        pass
    def sample_points(self,mask, num_points=16):
    # Get mask shape
        bz, H, W = mask.shape

        # Generate coordinates for sampling
        x_coordinates = torch.linspace(0, W-1, num_points).long()
        y_coordinates = torch.linspace(0, H-1, num_points).long()

        # Generate grid of coordinates
        x_grid, y_grid = torch.meshgrid(x_coordinates, y_coordinates)

        # Flatten the grid coordinates
        coordinates = torch.stack((y_grid.flatten(), x_grid.flatten()), dim=1)

        # Get mask values at coordinates
        mask_values = mask[:, y_grid.flatten(), x_grid.flatten()]

        # Threshold mask values to determine foreground or background
        labels = (mask_values > 0.5).float()

        # Reshape coordinates and labels
        coordinates = coordinates.unsqueeze(0).repeat(bz, 1, 1)
        # coordinates = coordinates.permute(0,2,1)
        labels = labels.view(bz, num_points * num_points)

        return coordinates, labels
    def clear_boundary(self,masks):
        boundary_size =5
        masks[:,:,:,:boundary_size, :] = 0
        masks[:,:,:,-boundary_size:, :] = 0
        masks[:,:,:,:, :boundary_size] = 0
        masks[:,:,:,:, -boundary_size:] = 0
        return masks
        
    def extract_central_point_coordinates(self,masks):
        boundary_size =10
        masks[:boundary_size, :] = 0
        masks[-boundary_size:, :] = 0
        masks[:, :boundary_size] = 0
        masks[:, -boundary_size:] = 0
        foreground_indices = torch.nonzero(masks > 0.5, as_tuple=False)

# Extract coordinates from indices
        foreground_coordinates = foreground_indices[:, [1, 0]]  # Swap x, y to get (y, x) format

        # Compute centroid of foreground coordinates
        centroid = torch.mean(foreground_coordinates.float(), dim=0)
        
        # Return centroid coordinates reshaped to [bz, 1, 2]
        return centroid.view(1, 1, 2)
    def optimization(self, label,lr):
        # for param_group in  self.optimizer.param_groups:
        #     param_group['lr'] = lr 
        self.optimizer.zero_grad()
        self.optimizer_s.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        self.set_requires_grad(self.VideoNets, True)
        self.set_requires_grad(self.VideoNets_S,True)

        # self.set_requires_grad(self.VideoNets_S,True)
        # self.set_requires_grad(self.resnet, True)
        out_logits = self.output.view(label.size(0), -1)
        bz,length = out_logits.size()

        label_mask_torch = torch.tensor(label_mask, dtype=torch.float32)
        label_mask_torch = label_mask_torch.repeat(bz, 1)
        label_mask_torch = label_mask_torch.to(self.device)

        self.loss = self.customeBCE(out_logits * label_mask_torch, label * label_mask_torch)
        # self.lossEa.backward(retain_graph=True)
        self.loss.backward( )

        self.optimizer.step()

        out_logits_s = self.output_s.view(label.size(0), -1)
        # bz,length = out_logits.size()

        # label_mask_torch = torch.tensor(label_mask, dtype=torch.float32)
        # label_mask_torch = label_mask_torch.repeat(bz, 1)
        # label_mask_torch = label_mask_torch.to(self.device)
        self.loss_s_v = self.customeBCE(out_logits_s * label_mask_torch, label * label_mask_torch)
        bz, ch, D, H, W = self.cam3D_s.size()

        valid_masks_repeated = self.slice_hard_label.repeat(1, 1, 1, H, W)
        predit_mask= self.cam3D_s * valid_masks_repeated
        target_mask= self.cam3D_target  * valid_masks_repeated
        # self.loss_s_pix = self.customeBCE(self.cam3D_s * valid_masks_repeated, self.binary_masks * valid_masks_repeated)
        self.loss_s_pix = self.customeBCE_mask(predit_mask , target_mask   )

        self.loss_s = self.loss_s_v  +self.loss_s_pix
        # self.set_requires_grad(self.VideoNets, False)
        self.loss_s.backward( )
        self.optimizer_s.step()
        self.lossDisplay = self.loss. data.mean()
        self.lossDisplay_s = self.loss_s. data.mean()

    def optimization_slicevalid(self):

        pass