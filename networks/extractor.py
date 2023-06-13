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


import torch
import torch.nn.functional as F
from config import cfg
from data_utils import *


def attn_cosine_sim(x, y, eps=1e-08):
    x = x[0]
    y = y[0]
    norm1 = x.norm(dim=2, keepdim=True)
    norm2 = y.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm2.permute(0,2,1), min=eps)
    sim_matrix = (x @ y.permute(0,2,1)) / factor
    return sim_matrix


class VitExtractor:
    QKV_KEY = 'qkv'
    ATTN_KEY = 'attn'
    BLOCK_KEY = 'block'
    PATCH_IMD_KEY = 'patch_imd'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, device):
        if 'dinov2' in model_name:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
        else:
            self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [11]
        self.layers_dict[VitExtractor.QKV_KEY] = [11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [11]
        for key in VitExtractor.KEY_LIST:
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 14

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768
        
    def extract_feat(self, inputs):
        features = []
        cls_tokens = []
        saliency_maps = []        
        for img in inputs:
            img = ((img.permute(0,2,3,1) - img_mean) / img_std).permute(0,3,1,2)
            height = self.get_height_patch_num(img.shape)
            width = self.get_width_patch_num(img.shape)
            t = self.get_patch_num(img.shape) # number of patches
            d = self.get_embedding_dim() # embedding_dim
            h = self.get_head_num() # number of heads
            # extract features
            self._register_hooks()
            self.model(img)
            feat = self.outputs_dict[VitExtractor.BLOCK_KEY][-1]
            attn = self.outputs_dict[VitExtractor.ATTN_KEY][-1]
            qkv = self.outputs_dict[VitExtractor.QKV_KEY][-1]
            self._clear_hooks()
            self._init_hooks_data()
            # class attention map
            head_idxs = [0,2,4,5]
            cls_attn_map = attn[:,head_idxs,0,1:].mean(dim=1)
            temp_min, temp_max = cls_attn_map.min(), cls_attn_map.max()
            cls_attn_map = (cls_attn_map - temp_min) / (temp_max - temp_min)
            saliency_maps.append(cls_attn_map[0])
            # extract queries, keys, values
            qkv = qkv.view(t,3,d).permute(1,0,2)
            q = qkv[0]
            k = qkv[1]
            v = qkv[2]
            features.append(F.normalize(k)[1:])
        return features, saliency_maps
    
    def extract_feat_hr(self, inputs):
        h, w = cfg.crop_size[1]//2, cfg.crop_size[0]//2
        input_patches = []
        for img in inputs:
            input_patches.append(img[...,:h,:w]) # top left
            input_patches.append(img[...,:h,w:]) # top right
            input_patches.append(img[...,h:,:w]) # bottom left
            input_patches.append(img[...,h:,w:]) # bottom right
        d = self.get_embedding_dim()
        ps = self.get_patch_size()
        feat, sal = self.extract_feat(input_patches)        
        feat = [f.view(h//ps, w//ps, d) for f in feat]
        feat = [torch.cat(feat[i:i+2], 1) for i in range(0,len(feat),2)] 
        feat = [torch.cat(feat[i:i+2], 0) for i in range(0,len(feat),2)]
        feat = [f.view(-1,d) for f in feat]        
        sal = [s.view(h//ps, w//ps) for s in sal]
        sal = [torch.cat(sal[i:i+2], 1) for i in range(0,len(sal),2)] 
        sal = [torch.cat(sal[i:i+2], 0) for i in range(0,len(sal),2)] 
        sal = [s.view(-1) for s in sal]
        return feat, sal
    