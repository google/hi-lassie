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


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
    
    
class PartMLP(nn.Module):
    def __init__(self, n_layers=6, d_in=3, d_out=3):
        super().__init__()
        self.n_layers = n_layers
        self.register_buffer("frequencies", 0.3*np.pi*torch.arange(1,n_layers+2))
        self.hidden_layers = nn.ModuleList([nn.Linear(d_in*2, d_in*2) for _ in range(n_layers)])
        self.output_layers = nn.ModuleList([nn.Linear(d_in*2, d_out, bias=False) for _ in range(n_layers)])
        for i in range(n_layers):
            nn.init.xavier_uniform_(self.output_layers[i].weight, gain=0.001)

    def freeze_layers(self, freeze_to):
        for i in range(freeze_to):
            self.hidden_layers[i].weight.requires_grad = False
            self.hidden_layers[i].bias.requires_grad = False
            self.output_layers[i].weight.requires_grad = False
            
    def freeze(self, freeze_to):
        for i in range(freeze_to):
            self.hidden_layers[i].weight.grad = None
            self.hidden_layers[i].bias.grad = None
            self.output_layers[i].weight.grad = None
            
    def positional_encoding(self, x):
        embed = (x[..., None] * self.frequencies)
        return torch.cat([embed.sin(), embed.cos()], dim=-2)
    
    def forward(self, x, stop_at=10):
        x = self.positional_encoding(x)
        out = 0
        hidden = x[...,0]
        for i in range(self.n_layers):
            hidden = x[...,i+1] * self.hidden_layers[i](hidden)
            out += self.output_layers[i](hidden)
            if i == stop_at:
                break
        return out
    