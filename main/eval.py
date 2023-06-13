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
import numpy as np
import torch
from argparse import ArgumentParser
from config import cfg
from dataloader import *
from data_utils import *
from model import *


def eval_model():
    print("========== Loading data of %s... ========== " % cfg.animal_class)
    num_imgs, inputs = load_data()
    
    print("========== Preparing model... ========== ")
    skeleton_path = osp.join(cfg.model_dir, 'skeleton_%s.json' % cfg.animal_class)
    model = Model(cfg.device, num_imgs, skeleton_path)
    model.load_model(osp.join(cfg.model_dir, '%s.pth'%cfg.animal_class))
    rasterizer = model.hard_renderer.renderer.rasterizer
    
    uvs, faces = model.meshes[3].get_uvs_and_faces()
    inputs['uvs'], inputs['faces'] = uvs, faces
    output_verts = []
    output_verts_2d = []
    for i in range(num_imgs):
        model.load_parts(osp.join(cfg.model_dir, '%s_part_%d.pth'%(cfg.animal_class, i)))
        outputs = model.forward(inputs, stop_at=10, text=None)
        output_verts.append(outputs['verts'])
        output_verts_2d.append(outputs['verts_2d'])
    
    print("========== Keypoint transfer evaluation... ========== ")
    pck = 0
    num_pairs = 0
    for i1 in range(num_imgs):
        for i2 in range(num_imgs):
            if i1 == i2:
                continue
            kps1 = inputs['kps_gt'][i1].cpu()
            kps2 = inputs['kps_gt'][i2].cpu()
            verts1 = output_verts_2d[i1].cpu().reshape(-1,2)
            verts2 = output_verts_2d[i2].cpu().reshape(-1,2)
            verts1_vis = get_visibility_map(output_verts[i1], faces, rasterizer).cpu()
            v_matched = find_nearest_vertex(kps1, verts1, verts1_vis)
            kps_trans = verts2[v_matched]
            valid = (kps1[:,2] > 0) * (kps2[:,2] > 0)
            dist = ((kps_trans - kps2[:,:2])**2).sum(1).sqrt()
            pck += ((dist <= 0.1) * valid).sum() / valid.sum()
            num_pairs += 1            
    pck /= num_pairs
    print('PCK=%.4f' % pck)
    
    if cfg.animal_class in ['horse', 'cow', 'sheep']:
        print("========== IOU evaluation... ==========")
        iou = 0
        for i in range(num_imgs):
            valid_parts = 0
            masks = get_part_masks(output_verts[i], faces, rasterizer).cpu()
            masks_gt = inputs['part_masks'][i,0].cpu()
            iou += mask_iou(masks>0, masks_gt>0)
        iou /= num_imgs
        print('Overall IOU = %.4f' % iou)
        
        
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cls', type=str, default='zebra', dest='cls')
    parser.add_argument('--inst', type=bool, default=False, dest='opt_instance')
    parser.add_argument('--idx', type=int, default=0, dest='instance_idx')
    args = parser.parse_args()
    cfg.set_args(args)
    
    with torch.no_grad():
        eval_model()
        