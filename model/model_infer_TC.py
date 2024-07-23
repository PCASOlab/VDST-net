import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.vision_transformer import vit_small, vit_base

from model.model_3dcnn_linear_TC import _VideoCNN
from model.model_3dcnn_linear_ST import _VideoCNN_S
from working_dir_root import learningR,learningR_res,SAM_pretrain_root,Load_feature,Weight_decay,Evaluation,config_root
from dataset.dataset import class_weights
 
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
        self.Vit_encoder = vit_small(8)
       
        self.Vit_encoder.load_state_dict(torch.load(config_root + "dino_deitsmall8_pretrain.pth"), strict=False)
        self.VideoNets = _VideoCNN()
        self.VideoNets_S = _VideoCNN_S()
        self.input_size = 1024
        resnet34 = models.resnet34(pretrained=True)
        self.gradcam = None
        
        # Remove the fully connected layers at the end
        partial = nn.Sequential(*list(resnet34.children())[0:-2])
        
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
        {'params': self.VideoNets.parameters(),'lr': learningR},
        {'params': self.Vit_encoder.parameters(),'lr': learningR*0.9}
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
 
    def optimization(self, label,Enable_ST = False):
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
        if Enable_ST:

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