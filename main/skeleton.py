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


from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import euler_angles_to_matrix
from geometry import *
from config import cfg


class Skeleton():
    def __init__(self, device, params_init):
        self.device = device
        joints = params_init['joints']
        self.joints = torch.tensor(joints).float().to(device)*2
        self.nj = len(joints)
        self.nb = self.nj -1
        self.euler_convention = ['YZX']*self.nb
        self.left_joints = torch.sign(self.joints[:,0])
        # self.sym_parts = torch.nonzero(self.joints[1:,0]).long()
        
        self.parent = params_init['joints_parent'] 
        self.bones_lev = {}
        self.bones_lev[0] = torch.arange(self.nb).long().to(device)
        self.bones_lev[1] = torch.tensor([i for i in range(self.nb) if self.parent[i+1] != 0]).long().to(device)
        self.bones_lev[2] = torch.tensor([i for i in range(self.nb) if self.parent[self.parent[i+1]] != 0]).long().to(device)

        self.joint_tree = torch.eye(self.nj).bool().to(device)
        self.bones = torch.zeros(2, self.nb).long().to(device)
        for i in range(1, self.nj):
            j1, j2 = self.parent[i], i
            self.bones[0,i-1] = j1
            self.bones[1,i-1] = j2
            i_back = self.nj - i
            j1, j2 = self.parent[i_back], i_back
            self.joint_tree[j1] = torch.logical_or(self.joint_tree[j1], self.joint_tree[j2])
        
        self.joints_sym = torch.zeros(self.nj).long().to(device)
        for i in range(self.nj):
            joint_flipped = self.joints[i].clone()
            joint_flipped[0] *= -1
            self.joints_sym[i] = ((self.joints - joint_flipped)**2).sum(1).argmin()
            
        self.init_bone_rot()

    def init_bone_rot(self):
        joints1 = self.joints[self.bones[0],:]
        joints2 = self.joints[self.bones[1],:]
        vec_bone = F.normalize(joints2 - joints1, p=2.0, dim=1) # nb x 3
        vec_bone = torch.where(vec_bone[:,1:2]>0, -vec_bone, vec_bone)
        vec_rest = torch.tensor([0,-1,0])[None,:].repeat(self.nb,1).float().to(self.device) # nb x 3
        self.bone_rot_init = get_mat_align(vec_rest, vec_bone)

    def transform_joints(self, rot, scale=None, symmetrize=False):
        bs = rot.shape[0]
        joints = self.joints.clone()
        # scaling
        if scale is not None:
            scale = (scale + scale[self.joints_sym[1:]-1])*0.5
            for i in range(1, self.nj):
                offset = (joints[i,:] - joints[self.parent[i],:])[None,:] * scale[i-1].clip(-0.5,1.5)
                joints[self.joint_tree[i]] += offset
        # rotation
        results_rot = [torch.eye(3)[None].repeat(bs,1,1).float().to(self.device)]
        results_trans = [joints[None,0].repeat(bs,1)]
        for i in range(1, self.nj):
            rot_mat = euler_angles_to_matrix(rot[:,i,:], self.euler_convention[i-1])
            rot_parent, trans_parent = results_rot[self.parent[i]], results_trans[self.parent[i]]
            joint_rel_trans = (joints[i] - joints[self.parent[i]])[:,None]
            results_rot.append(torch.bmm(rot_parent, rot_mat))
            results_trans.append(torch.matmul(results_rot[i], joint_rel_trans)[:,:,0] + trans_parent)
        joints = torch.stack(results_trans, 1)
        joints_rot = torch.stack(results_rot, 1)[:,1:]
        # symmetrize
        if symmetrize:
            joints = self.symmetrize_joints(joints)
        return joints, joints_rot

    def symmetrize_joints(self, joints):
        joints_ref = joints.clone()
        joints_ref[:,:,0] *= -1
        return (joints + joints_ref[:,self.joints_sym,:])/2
