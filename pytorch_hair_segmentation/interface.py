import cv2
import numpy as np
import torch
import time
import os
import sys
import argparse
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from networks import get_network
from data import get_loader
import torchvision.transforms as std_trnsf
from utils import joint_transforms as jnt_trnsf
from utils.metrics import MultiThresholdMeasures

def has_img_ext(fname):
    ext = os.path.splitext(fname)[1]
    return ext in ('.jpg', '.jpeg', '.png')
class HairSegment:
    def __init__(self,use_gpu) -> None:
        ckpt_dir = os.path.join(os.path.dirname(__file__),"../checkpoints/pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth")
        network = 'pspnet_resnet101'.lower()
        self.device = 'cuda' if use_gpu else 'cpu'

        assert os.path.exists(ckpt_dir)
        # prepare network with trained parameters
        self.net = get_network(network).to(self.device)
        state = torch.load(ckpt_dir)
        self.net.load_state_dict(state['weight'])
        
        self.test_image_transforms = std_trnsf.Compose([
            std_trnsf.ToTensor(),
            std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
    def get_segment(self,img,reqCode):
        # prepare images
        with torch.no_grad():
            data = self.test_image_transforms(img)
            data = torch.unsqueeze(data, dim=0)
            self.net.eval()
            data = data.to(self.device)

            # inference
            
            logit = self.net(data)

            # prepare mask
            pred = torch.sigmoid(logit.cpu())[0][0].data.numpy()
            mh, mw = data.size(2), data.size(3)
            mask = pred >= 0.5

            mask_n = np.zeros((mh, mw, 3))
            mask_n[:,:,0] = 255
            mask_n[:,:,0] *= mask

            image_n = np.array(img)
            image_n = cv2.cvtColor(image_n, cv2.COLOR_RGB2BGR)
            # discard padded area
            ih, iw, _ = image_n.shape

            delta_h = mh - ih
            delta_w = mw - iw

            top = delta_h // 2
            bottom = mh - (delta_h - top)
            left = delta_w // 2
            right = mw - (delta_w - left)

            mask_n = mask_n[top:bottom, left:right, :]

            # addWeighted
            # image_n = image_n * 0.5 +  mask_n * 0.5

            # write overlay image
            # cv2.imwrite(os.path.join(".", reqCode),image_n)
            # cv2.imwrite(os.path.join(".","mask.png"),mask_n[...,0])

            return mask_n[...,0]