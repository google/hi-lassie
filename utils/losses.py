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
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from config import cfg


# silhouette loss
def sil_loss(verts, faces, part_masks, renderer, valid=None):
    mask_gt = (part_masks[:,0,:,:] > 0).float()
    sil = renderer.render(verts, faces)[:,:,:,3]
    loss = (sil - mask_gt)**2
    return loss.mean()

# semantic consistency loss
def feature_loss(feat_verts, verts, feat_img, masks, w_xy=1):
    bs, nv, df, hw = verts.shape[0], verts.shape[2], feat_img.shape[1], feat_img.shape[-1]
    # image features (feat_img: bs x d x 64 x 64)
    grid = torch.arange(hw).float().to(cfg.device)
    grid_y, grid_x = torch.meshgrid(grid, grid, indexing='ij')
    xy_img = torch.stack([grid_x, grid_y], 0)[None].repeat(bs,1,1,1)/(hw-1) * w_xy
    pts_img = torch.cat([xy_img, (masks>0).float(), feat_img/np.sqrt(df)], 1)
    pts_img = pts_img.view(bs,df+3,-1).permute(0,2,1)
    # vertex features (feat_vert: nb x nv x d)
    xy_verts = verts.permute(0,3,1,2).view(bs,2,-1).permute(0,2,1) * w_xy
    sal_verts = torch.ones_like(xy_verts[:,:,:1])
    feat_verts = feat_verts.permute(1,0)[None,:,:,None].repeat(bs,1,1,nv).view(bs,df,-1).permute(0,2,1).detach()
    pts_verts = torch.cat([xy_verts, sal_verts, feat_verts/np.sqrt(df)], 2)
    # chamfer distance
    knn1 = knn_points(pts_img, pts_verts, K=1) # for each pixel, find corr. vertex
    knn2 = knn_points(pts_verts, pts_img, K=1) # for each vertex, find corr. pixel
    dist1 = (knn1.dists[...,0] * pts_img[:,:,2]).mean(1)
    dist2 = knn2.dists[...,0].mean(1)
    return dist1.mean() + dist2.mean() * 0.1

# part center loss
def part_center_loss(joints, part_centers, part_mapping, bones):
    bs = joints.shape[0]
    loss = 0
    for i in range(part_centers.shape[1]):
        if (part_mapping == i).sum() > 0:
            js1 = joints[:,bones[0],:][:,part_mapping==i,:]
            js2 = joints[:,bones[1],:][:,part_mapping==i,:]
            center = ((js1 + js2)/2).mean(1)
            if i == part_mapping[1] or i == part_mapping[-1]:
                loss += ((center - part_centers[:,i,:2])**2 * part_centers[:,i,2:]).sum() * 0.1
    return loss / bs

# bone rotation prior
def canonical_prior_loss(joints_can, joints_sym, joints_parent, left_joints):
    bs = joints_can.shape[0]
    loss = 0
    loss += joints_can[:,:,0].mean()**2
    loss += ((joints_can[:,:,1:] - joints_can[:,joints_sym,1:])**2).mean()
    loss += ((joints_can[:,:,0] + joints_can[:,joints_sym,0])**2).mean()
    # loss += ((F.relu(0.05-joints_can[:,:,0])**2) * (left_joints > 0).float()).sum()
    # loss += ((F.relu(0.05+joints_can[:,:,0])**2) * (left_joints < 0).float()).sum()
    return loss / bs

# pose deviation from resting pose
def pose_prior_loss(bone_scale, bone_rot, part_codes):
    nb = bone_scale.shape[0]
    loss = 0
    # bone scale
    loss += (bone_scale**2).sum() * 0.001
    # bone rot
    loss += (bone_rot[:,:,1]**2).sum() * 0.05 # constrain raw-axis rotation
    loss += (bone_rot[:,:,2]**2).sum() * 0.02 # constrain pitch-axis rotation
    loss += (bone_rot[:,0,2]**2).sum() * 0.05 # constrain pitch-axis rotation
    # part codes
    # loss += ((part_codes[1:,0]-0.05)**2).sum() * 0.05
    # loss += ((part_codes[0:,1]-1.1)**2).sum() * 0.05
    # loss += (F.relu(0.05-part_codes[...,0])**2).sum()
    # loss += (F.relu(1.1-part_codes[...,1])**2).sum()
    return loss / nb

# general pose prior
def global_prior_loss(global_rot):
    bs = global_rot.shape[0]
    loss = 0
    loss += (global_rot[:,0]**2).sum()
    loss += (global_rot[:,1]**2).sum() * 0.1
    # loss += (F.relu(-np.pi/2 + global_rot[:,2])**2).sum()
    # loss += (F.relu(-np.pi/2 - global_rot[:,2])**2).sum()
    return loss / bs

# shape prior loss
def shape_prior_loss(f_parts, uvs):
    nb = len(f_parts)
    loss = 0
    for i, f in enumerate(f_parts):
        deform = f(uvs[None,:,:])
        loss += (deform**2).mean() * 0.01
    return loss / nb

# instance prior loss
def instance_prior_loss(f_parts, uvs, f_instance, stop_at):
    nb = len(f_parts)
    loss = 0
    for i, f in enumerate(f_parts):
        with torch.no_grad():
            base_shape = f(uvs[None,:,:], stop_at=f_instance-1)
        loss += ((f(uvs[None,:,:], stop_at=stop_at) - base_shape)**2).mean()
    return loss / nb

def edge_loss(verts, faces):
    bs, nb, nv = verts.shape[:3]
    mesh = Meshes(verts=verts.reshape(-1,nv,3), faces=faces[None,:,:].repeat(bs*nb,1,1))
    return mesh_edge_loss(mesh)

def normal_loss(verts, faces):
    bs, nb, nv = verts.shape[:3]
    mesh = Meshes(verts=verts.reshape(-1,nv,3), faces=faces[None,:,:].repeat(bs*nb,1,1))
    return mesh_normal_consistency(mesh)

def laplacian_loss(verts, faces):
    bs, nb, nv = verts.shape[:3]
    mesh = Meshes(verts=verts.reshape(-1,nv,3), faces=faces[None,:,:].repeat(bs*nb,1,1))
    return mesh_laplacian_smoothing(mesh, method="uniform")

def inflation_loss(verts, faces, eps=1e-5):
    v0 = verts[:, :, faces[:,0], :].reshape(-1,3)
    v1 = verts[:, :, faces[:,1], :].reshape(-1,3)
    v2 = verts[:, :, faces[:,2], :].reshape(-1,3)
    norms = torch.cross(v1 - v0, v2 - v0)
    norms = F.normalize(norms, dim=-1)
    v0_t = (v0 + norms).detach()
    v1_t = (v1 + norms).detach()
    v2_t = (v2 + norms).detach()
    loss = (((v0_t - v0)**2).sum(1) + eps).sqrt().mean()
    loss += (((v1_t - v1)**2).sum(1) + eps).sqrt().mean()
    loss += (((v2_t - v2)**2).sum(1) + eps).sqrt().mean()
    return loss

def mse_loss(text1, text2, valid=None):
    loss = (text1 - text2)**2
    if valid is not None:
        loss *= valid
    return loss.mean()
