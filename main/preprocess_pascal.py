# Copyright 2023 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os.path as osp
import glob
import cv2
import numpy as np
import json
from scipy.io import loadmat
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from argparse import ArgumentParser
from config import cfg
from data_utils import *
from extractor import *
from clustering import *


def preprocess_data():        
    print("Reading images and annotations of %s..." % cfg.animal_class)
    images = []
    bbox_gt = []
    kps_gt = []
    masks_gt = []
    
    img_list = osp.join(cfg.pascal_img_set_dir, '%s.txt'%cfg.animal_class)
    with open(img_list, 'r') as f:
        img_files = [img_file.replace('\n','') for img_file in f.readlines()]
        
    for i, img in enumerate(img_files):
        img_id = img.split('/')[-1].replace('.jpg','')
        ann_file = osp.join(cfg.pascal_ann_dir, img_id + '.mat')
        ann = loadmat(ann_file)
        obj = ann['anno'][0,0]['objects'][0,0]
        parts = obj["parts"]
        
        img = cv2.imread(osp.join(cfg.pascal_img_dir, img))[:,:,2::-1]/255.
        part_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        part_centers = np.zeros((16,3))
        keypoints = np.zeros((14,3))
        
        for j in range(parts.shape[1]):
            part = parts[0,j]
            part_name = part["part_name"][0]
            mask = part["mask"]
            part_idx = part_indices[part_name]
            part_mask[mask > 0] = part_idx
            center, left, right, top, bottom = find_corners(mask)
            part_centers[part_idx-1,:] = center[0], center[1], 1
            if part_name == 'muzzle':
                keypoints[kp_indices[part_name],:] = (center + bottom)/2
            elif part_name == 'tail':
                keypoints[kp_indices[part_name],:] = top
            elif part_name in ['rfuleg', 'rbuleg', 'lfuleg', 'lbuleg']:
                keypoints[kp_indices[part_name],:] = bottom
            elif part_name in ['rflleg', 'rblleg', 'lflleg', 'lblleg']:
                keypoints[kp_indices[part_name],:] = bottom 
            elif part_name in ['leye', 'reye', 'lear', 'rear']:
                keypoints[kp_indices[part_name],:] = center
        
        coords_y, coords_x = np.where(part_mask > 0)
        left = np.min(coords_x)
        top = np.min(coords_y)
        width = np.max(coords_x) - left
        height = np.max(coords_y) - top
        bb = process_bbox(left, top, width, height)
        bbox_gt.append(bb)
        
        keypoints[:,0] = ((keypoints[:,0] - bb[0]) / bb[2]) * keypoints[:,2]
        keypoints[:,1] = ((keypoints[:,1] - bb[1]) / bb[3]) * keypoints[:,2]
        kps_gt.append(torch.tensor(keypoints).float().to(cfg.device))
        
        img = crop_and_resize(img, bb, cfg.crop_size, rgb=True)
        part_mask = crop_and_resize(part_mask, bb, cfg.input_size, rgb=False)
        images.append(img)
        masks_gt.append(part_mask)
    
    print("Extracting DINO features...")
    extractor = VitExtractor(cfg.dino_model, cfg.device)        
    images = [F.interpolate(img, cfg.crop_size, mode='bilinear', align_corners=False) for img in images]
    with torch.no_grad():
        features, saliency = extractor.extract_feat_hr(images)

    print("Clustering DINO features...")
    masks_vit, part_centers, centroids = cluster_features(features, saliency, images)
    
    print("Extracting low-res DINO features...")
    images = [F.interpolate(img, cfg.input_size, mode='bilinear', align_corners=False) for img in images]
    with torch.no_grad():
        features, saliency = extractor.extract_feat(images)

    print("Clustering low-res features...")
    masks_fg = [F.interpolate((m>0).float(), cfg.input_size, mode='nearest') for m in masks_vit]
    masks_vit, part_centers, centroids = cluster_features(features, saliency, images, masks_fg)
    
    print("Collecting input batch...")
    inputs = {}
    inputs['images'] = F.interpolate(torch.cat(images, 0), cfg.input_size, mode='bilinear', align_corners=False)
    inputs['masks'] = F.interpolate(torch.cat(masks_vit, 0), cfg.input_size, mode='nearest')
    inputs['masks_gt'] = F.interpolate(torch.cat(masks_gt, 0), cfg.input_size, mode='nearest')
    inputs['masks_lr'] = F.interpolate(torch.cat(masks_vit, 0), (cfg.hw,cfg.hw), mode='nearest')
    inputs['kps_gt'] = torch.stack(kps_gt, 0)
    inputs['part_cent'] = torch.stack(part_centers, 0)
        
    # Reduce feature dimension
    d = extractor.get_embedding_dim()
    feat_img = torch.stack([k.permute(1,0).view(d,cfg.hw,cfg.hw) for k in features], 0)
    feat_sal = feat_img.permute(0,2,3,1)[inputs['masks_lr'][:,0]>0]
    _, _, V = torch.pca_lowrank(feat_sal, q=cfg.d_feat, center=True, niter=2)
    feat_img = feat_img.permute(1,0,2,3).reshape(d,-1).permute(1,0)
    inputs['feat_img'] = torch.matmul(feat_img, V).permute(1,0).view(cfg.d_feat,-1,cfg.hw,cfg.hw).permute(1,0,2,3)
    inputs['feat_part'] = torch.matmul(centroids, V)
        
    for i in range(len(images)):
        img = inputs['images'][i].permute(1,2,0)
        mask = inputs['masks'][i].permute(1,2,0).cpu().numpy()
        cmask = part_mask_to_image(mask[:,:,0], part_colors)
        save_img('proc_%d.png'%i, img2np(img))
        save_img('mask_vit_%d.png'%i, cmask)
        
    for k in inputs:
        if k == 'images':
            np.save(osp.join(cfg.input_dir, k+'.npy'), (inputs[k].cpu().numpy()*255.).astype(np.uint8))
        elif k in ['masks', 'masks_lr', 'masks_gt']:
            np.save(osp.join(cfg.input_dir, k+'.npy'), inputs[k].cpu().numpy().astype(np.uint8))
        else:
            np.save(osp.join(cfg.input_dir, k+'.npy'), inputs[k].cpu().numpy())
    
    return len(images)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cls', type=str, default='zebra', dest='cls')
    parser.add_argument('--inst', type=bool, default=False, dest='opt_instance')
    parser.add_argument('--idx', type=int, default=0, dest='instance_idx')
    args = parser.parse_args()
    cfg.set_args(args)
        
    num_imgs = preprocess_data()
    print("Finished preprocessing %d images." % num_imgs)
    