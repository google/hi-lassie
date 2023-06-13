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
from pytorch3d.utils import ico_sphere
from pytorch3d.transforms import axis_angle_to_matrix


class Mesh:
    def __init__(self, device, lev=1):
        self.lev = lev
        template_mesh = ico_sphere(lev, device)
        self.faces = template_mesh.faces_list()[0]
        self.verts = template_mesh.verts_list()[0]
        self.nv = self.verts.shape[0]
        self.nf = self.faces.shape[0]
        self.base_shape = self.verts.clone()
        
    def get_uvs_and_faces(self, gitter=False):
        uvs = self.verts.clone()
        faces = self.faces.clone()
        if gitter:
            uvs += torch.randn_like(uvs)*1e-2 / (2**self.lev)
        return uvs, faces
        
        
def get_rot_mat(azimuth=0, gitter=0):
    angle = torch.zeros(3).float()
    angle[0] += np.random.randn()*gitter
    angle[1] += azimuth * np.pi/180
    return axis_angle_to_matrix(angle)


def get_mat_align(vec1, vec2):
    bs = vec1.shape[0]
    v = torch.cross(vec1, vec2, dim=1) # nb x 3
    c = (vec1 * vec2).sum(1)[:,None,None] # nb x 1 x 1
    V = torch.zeros(bs,3,3).to(vec1.device)
    V[:,0,1] += -v[:,2]
    V[:,0,2] += v[:,1]
    V[:,1,0] += v[:,2]
    V[:,1,2] += -v[:,0]
    V[:,2,0] += -v[:,1]
    V[:,2,1] += v[:,0]
    rot = torch.eye(3)[None,:,:].repeat(bs,1,1).to(vec1.device)
    rot += V + torch.bmm(V,V)*(1/(1+c))
    return rot
    