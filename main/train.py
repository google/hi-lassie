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
from argparse import ArgumentParser
from config import cfg
from dataloader import *
from model import *


def train_model():        
    print("========== Loading data of %s... ========== " % cfg.animal_class)
    num_imgs, inputs = load_data()

    print("========== Preparing model... ========== ")
    skeleton_path = osp.join(cfg.model_dir, 'skeleton_%s.json' % cfg.animal_class)
    model = Model(cfg.device, num_imgs, skeleton_path)

    print("========== 3D optimization... ========== ")
    if cfg.opt_instance:
        model.load_model(osp.join(cfg.model_dir, '%s.pth'%cfg.animal_class), freeze_to=cfg.f_instance)
        model.optimize_instance(inputs)
        model.save_parts(osp.join(cfg.model_dir, '%s_part_%d.pth'%(cfg.animal_class, cfg.instance_idx)))
    else:
        model.train(inputs)
        model.save_model(osp.join(cfg.model_dir, '%s.pth'%cfg.animal_class))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cls', type=str, default='zebra', dest='cls')
    parser.add_argument('--inst', type=bool, default=False, dest='opt_instance')
    parser.add_argument('--idx', type=int, default=0, dest='instance_idx')
    args = parser.parse_args()
    cfg.set_args(args)

    train_model()
