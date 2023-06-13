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


import sys
import os
import os.path as osp
import torch


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


class Config:
    ## inputs
    dino_model = 'dino_vits8' # 'dino_vits8' / 'dinov2_vits14'
    ps = 8 if dino_model == 'dino_vits8' else 14
    crop_size = (ps*128, ps*128)
    input_size = (384, 384)
    hw = 64
    
    # optimization settings
    use_crf = True
    opt_instance = False
    instance_idx = 0
    n_clusters = 8
    d_latent = 64
    d_feat = 20
    f_instance = 5

    ## directory
    curr_dir = osp.dirname(osp.abspath(__file__))
    root_dir = osp.join(curr_dir, '..')
    data_root = osp.join(root_dir, 'data')
    lassie_img_dir = osp.join(data_root, 'lassie', 'images')
    lassie_ann_dir = osp.join(data_root, 'lassie', 'annotations')
    pascal_img_dir = '/hdd1/hank/JPEGImages/'
    pascal_ann_dir = '/hdd1/hank/pascal-parts/Annotations_Part/'
    pascal_img_set_dir = '/hdd1/hank/pascal-parts/image_sets/'
    
    device = torch.device("cuda")
    print('>>> Using device:', device)

    def set_args(self, args):
        self.animal_class = args.cls
        self.opt_instance = args.opt_instance
        self.instance_idx = args.instance_idx 
        self.input_dir = osp.join(self.data_root, 'preprocessed', self.animal_class)
        self.output_dir = osp.join(self.root_dir, 'results', self.animal_class)
        self.model_dir = osp.join(self.root_dir, 'model_dump')            
        make_folder(self.input_dir)
        make_folder(self.output_dir)
        make_folder(self.model_dir)
        

cfg = Config()

add_pypath(cfg.root_dir)
add_pypath(osp.join(cfg.root_dir, 'networks'))
add_pypath(osp.join(cfg.root_dir, 'utils'))
