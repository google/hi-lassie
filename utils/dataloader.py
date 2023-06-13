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
import numpy as np
import torch
import torch.nn.functional as F
from config import cfg
    
    
def load_data():
    inputs = {}
    for np_file in glob.glob(osp.join(cfg.input_dir, '*.npy')):
        k = np_file.split('/')[-1].split('.')[0]
        inputs[k] = torch.from_numpy(np.load(np_file)).float().to(cfg.device)
        if k == 'images':
            num_imgs = inputs[k].shape[0]
        if cfg.opt_instance and k != 'feat_part':
            inputs[k] = inputs[k][cfg.instance_idx,None]
    inputs['images'] /= 255.       
    inputs['images'] = F.interpolate(inputs['images'], cfg.input_size, mode='bilinear', align_corners=False)
    inputs['masks'] = F.interpolate(inputs['masks'], cfg.input_size, mode='nearest')
    return num_imgs, inputs
    