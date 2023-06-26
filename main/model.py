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
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d
from pytorch3d.transforms import euler_angles_to_matrix
from config import cfg
from data_utils import *
from geometry import *
from extractor import *
from clustering import *
from skeleton import *
from part_mlp import *
from rendering import *
from losses import *


class Model(nn.Module):
    def __init__(self, device, num_imgs, skeleton_path):
        super().__init__()
        self.device = device
        self.num_imgs = num_imgs

        # skeleton and part mapping
        skeleton_params = json.load(open(skeleton_path, 'r'))
        self.skeleton = Skeleton(device, skeleton_params)
        cfg.nb = self.skeleton.nb

        # rendering
        self.sil_renderer = Renderer(device, shader='soft_sil', light=None)
        self.hard_renderer = Renderer(device, shader='hard_phong', light='ambient')
        self.hard_shaded_renderer = Renderer(device, shader='hard_phong', light='point')

        # shared parameters
        self.feat_verts = torch.zeros(cfg.nb, cfg.d_feat).float().to(device)
        self.part_mapping = torch.tensor(skeleton_params['part_dino']).long().to(device)
        self.rot_id = torch.zeros(3).float().to(device)
        self.bone_rot_rest = nn.Parameter(self.rot_id[None,:].repeat(cfg.nb,1))
        self.bone_scale = nn.Parameter(torch.zeros(cfg.nb).float().to(device))
        self.f_parts = nn.ModuleList([PartMLP(n_layers=10, d_in=3, d_out=3).to(device) for i in range(cfg.nb)])
        part_scale_xz = torch.tensor(skeleton_params['part_scale_xz']).float().to(device)
        part_scale_y = torch.tensor(skeleton_params['part_scale_y']).float().to(device)
        part_codes = torch.stack([part_scale_xz, part_scale_y], 1)
        part_codes = (part_codes + part_codes[self.skeleton.joints_sym[1:]-1])*0.5
        self.part_codes = nn.Parameter(part_codes)

        # instance-specific parameters
        global_rot = self.rot_id[None,:].repeat(num_imgs,1)
        global_rot[:,2] = torch.tensor(skeleton_params['azimuth']).float().to(device)
        self.global_rot = nn.Parameter(global_rot)
        self.global_trans = nn.Parameter(-self.skeleton.joints[:,:2].mean(0)[None].repeat(num_imgs,1))
        self.global_scale = nn.Parameter(torch.ones(num_imgs).float().to(device))
        self.bone_rot = nn.Parameter(self.rot_id[None,None,:].repeat(num_imgs,cfg.nb,1))

        # mesh vertices and symmetry
        self.meshes = {
            1: Mesh(device, 1), # 42
            2: Mesh(device, 2), # 162
            3: Mesh(device, 3), # 642
            4: Mesh(device, 4), # 2562
            5: Mesh(device, 5), # 10242
        }
        self.nv2lev = {}
        self.verts_sym = {}
        for k in [1,2,3,4,5]:
            nv, uvs = self.meshes[k].nv, self.meshes[k].verts.clone()
            self.nv2lev[nv] = k
            self.verts_sym[nv] = torch.zeros(cfg.nb*nv).long().to(device)
            verts = self.transform_verts(uvs, self.skeleton.joints[None], deform=False)
            verts = verts.reshape(cfg.nb*nv,3)
            verts_ref = verts.clone()
            verts_ref[:,0] *= -1
            for i in range(cfg.nb*nv):
                self.verts_sym[nv][i] = ((verts_ref - verts[i,:])**2).sum(1).argmin()

    def get_instance_params(self, idx=None):
        if cfg.opt_instance:
            global_scale = self.global_scale[cfg.instance_idx,None]
            global_trans = self.global_trans[cfg.instance_idx,None]
            global_rot = self.global_rot[cfg.instance_idx,None]
            bone_rot = self.bone_rot[cfg.instance_idx,None]
        elif idx is not None:
            global_scale = self.global_scale[idx,None]
            global_trans = self.global_trans[idx,None]
            global_rot = self.global_rot[idx,None]
            bone_rot = self.bone_rot[idx,None]
        else:
            global_scale = self.global_scale
            global_trans = self.global_trans
            global_rot = self.global_rot
            bone_rot = self.bone_rot
        global_rot = euler_angles_to_matrix(global_rot, 'ZXY')
        return global_scale*0.1+1, global_trans, global_rot, bone_rot

    def freeze_parts(self, lev=0):
        self.bone_rot.grad[:,:,0] = 0
        if lev > 0 and lev < 3:
            self.bone_rot.grad[:,self.skeleton.bones_lev[lev]] = 0
            self.bone_scale.grad[self.skeleton.bones_lev[lev]] = 0
            self.part_codes.grad[self.skeleton.bones_lev[lev]] = 0
            
    def freeze_instances_except(self, idx=0):
        for i in range(self.bone_rot.shape[0]):
            if i != idx:
                self.bone_rot.grad[i] = 0

    def update_feat_verts(self, verts_2d, feat_img, masks):
        with torch.no_grad():
            feat_verts = F.grid_sample(feat_img, verts_2d*2-1, align_corners=True).permute(0,2,3,1)
            sal_verts = F.grid_sample((masks>0).float(), verts_2d*2-1, align_corners=True).permute(0,2,3,1)
            feat_verts_avg = (feat_verts * sal_verts).sum(0) / (sal_verts.sum(0) + 1e-6)
            self.feat_verts = 0.9 * self.feat_verts + 0.1 * feat_verts_avg.mean(1)

    def update_bone_rot_rest(self):
        bs = 1 if cfg.opt_instance else self.num_imgs
        with torch.no_grad():
            mean_bone_rot = self.bone_rot.mean(0)
            self.bone_rot -= mean_bone_rot[None]
            self.bone_rot_rest += mean_bone_rot

    def symmetrize_verts(self, verts):
        nv = verts.shape[-2]
        verts_sym = self.verts_sym[nv]
        verts = verts.reshape(-1,cfg.nb*nv,3)
        verts_ref = verts.clone()
        verts_ref[:,:,0] *= -1
        return ((verts + verts_ref[:,verts_sym,:])/2).view(-1,cfg.nb,nv,3)

    def transform_verts(self, uvs, joints, bone_rot=None, deform=True, stop_at=10):
        bs, nv = joints.shape[0], uvs.shape[0]
        joint1 = joints[:,self.skeleton.bones[0],:]
        joint2 = joints[:,self.skeleton.bones[1],:]
        bone_trans = (joint1 + joint2)*0.5
        bone_scale = ((joint1 - joint2)**2).sum(2).sqrt()*0.5
        verts = uvs[None].repeat(cfg.nb,1,1)
        if deform:
            verts += torch.cat([f(uvs[None], stop_at) for f in self.f_parts], 0) # P x V x 3
        s_xz = self.part_codes[:,None,0].clamp(0.05, 1.5)*2
        s_y = self.part_codes[:,None,1].clamp(1.0, 2.0)
        verts[:,:,0] *= s_xz
        verts[:,:,2] *= s_xz
        verts[:,:,1] *= s_y
        verts *= bone_scale[0,:,None,None]
        verts = torch.bmm(self.skeleton.bone_rot_init, verts.permute(0,2,1)).permute(0,2,1) # P x V x 3
        if deform:
            verts = self.symmetrize_verts(verts)[0]
        verts = verts[None].repeat(bs,1,1,1).permute(0,1,3,2).view(-1,3,nv) # B*P x 3 x V
        if bone_rot is not None:
            verts = torch.bmm(bone_rot.reshape(-1,3,3), verts) # bone rotation
        verts = verts.view(bs,cfg.nb,3,nv).permute(0,1,3,2) + bone_trans[:,:,None,:] # bone translation
        return verts # B x P x V x 3
    
    def global_transform(self, x, rot, trans=None, scale=None):
        bs = rot.shape[0]
        x = torch.bmm(rot, x.permute(0,2,1)).permute(0,2,1)
        if scale is not None:
            x *= scale[:,None,None]
        if trans is not None:
            x[...,:2] += trans[:,None,:]
        return x
    
    def get_view(self, verts, azimuth=None, gitter=0.0):
        bs = verts.shape[0]
        center = (verts.max(2)[0].max(1)[0] + verts.min(2)[0].min(1)[0])*0.5
        verts -= center[:,None,None,:]
        scale = (verts**2).sum(3).max(2)[0].max(1)[0].sqrt()
        global_rot = get_rot_mat(azimuth, gitter)[None].repeat(bs,1,1).to(self.device)
        verts = self.global_transform(verts.reshape(bs,-1,3), global_rot)
        verts = verts / scale[:,None,None].detach()
        return verts.reshape(bs,cfg.nb,-1,3)

    def get_resting_shape(self, inputs, deform=True, stop_at=10):
        bs = 1
        root_rot = self.rot_id[None,None,:].repeat(bs,1,1) # bs x 1 x 3
        rot = torch.cat([root_rot, self.bone_rot_rest[None,:,:]], 1) # bs x nj x 3
        joints_can, joints_rot = self.skeleton.transform_joints(rot, scale=self.bone_scale)
        verts = self.transform_verts(inputs['uvs'], joints_can, joints_rot, deform, stop_at)
        return verts
        
    def forward(self, inputs, deform=True, stop_at=10, text=None):
        bs = 1 if cfg.opt_instance else self.num_imgs
        global_scale, global_trans, global_rot, bone_rot = self.get_instance_params()
        root_rot = self.rot_id[None,None,:].repeat(bs,1,1) # B x 1 x 3
        rot = torch.cat([root_rot, bone_rot + self.bone_rot_rest[None,:,:]], 1) # B x J x 3
        joints_can, joints_rot = self.skeleton.transform_joints(rot, scale=self.bone_scale)
        verts_can = self.transform_verts(inputs['uvs'], joints_can, joints_rot, deform, stop_at)
        joints = self.global_transform(joints_can, global_rot, global_trans, global_scale)
        verts = self.global_transform(verts_can.reshape(bs,-1,3), global_rot, global_trans, global_scale)
        verts_2d = self.sil_renderer.project(verts).view(bs,cfg.nb,-1,3)
        joints_2d = self.sil_renderer.project(joints)        
        outputs = {}
        outputs['verts'] = verts.reshape(bs,cfg.nb,-1,3)
        outputs['joints'] = joints
        outputs['verts_can'] = verts_can
        outputs['joints_can'] = joints_can
        outputs['verts_2d'] = verts_2d[...,:2] / cfg.input_size[0]
        outputs['joints_2d'] = joints_2d[...,:2] / cfg.input_size[0]
        outputs['verts_color'] = self.get_texture(inputs, outputs, text)
        return outputs

    def get_texture(self, inputs, outputs, mode=None):
        if mode is None:
            return None
        nv = inputs['uvs'].shape[0]
        verts_sym = self.verts_sym[nv]
        bs = inputs['images'].shape[0]
        verts_color = F.grid_sample(inputs['images'], outputs['verts_2d']*2-1, 
                                    align_corners=False).view(bs,3,-1).permute(0,2,1)
        verts_combined = outputs['verts'].permute(0,3,1,2).view(bs,3,-1).permute(0,2,1)
        visible = verts_combined[:,:,2] >= verts_combined[:,verts_sym,2]
        verts_color = torch.where(visible[:,:,None], verts_color, verts_color[:,verts_sym,:])
        verts_color = verts_color.permute(0,2,1).view(bs,3,cfg.nb,nv).permute(0,2,3,1)
        return verts_color # B x P x V x C

    def calculate_losses(self, inputs, outputs, losses, weights, params):
        uvs, faces = self.meshes[params['mesh_lev']].get_uvs_and_faces(gitter=True)
        
        for k in weights:
            loss = 0            
            if k == 'feat' and weights[k] > 0:
                loss = feature_loss(self.feat_verts, outputs['verts_2d'], inputs['feat_img'], 
                                    inputs['masks_lr'], params['w_xy'])
            elif k == 'sil' and weights[k] > 0:
                loss = sil_loss(outputs['verts'], inputs['faces'], inputs['masks'], self.sil_renderer)
            elif k == 'part_cent' and weights[k] > 0:
                loss = part_center_loss(outputs['joints_2d'], inputs['part_cent'],
                                        self.part_mapping, self.skeleton.bones)
            elif k == 'can_prior' and weights[k] > 0:
                loss = canonical_prior_loss(outputs['joints_can'], self.skeleton.joints_sym, 
                                            self.skeleton.parent, self.skeleton.left_joints)
            elif k == 'global_prior' and weights[k] > 0:
                loss = global_prior_loss(self.global_rot)
            elif k == 'pose_prior' and weights[k] > 0:
                loss = pose_prior_loss(self.bone_scale, self.bone_rot, self.part_codes)
            elif k == 'shape_prior' and weights[k] > 0:
                loss = shape_prior_loss(self.f_parts, uvs)
            elif k == 'instance' and weights[k] > 0:
                loss = instance_prior_loss(self.f_parts, uvs, cfg.f_instance, params['stop_at'])
            elif k == 'norm' and weights[k] > 0:
                loss = normal_loss(outputs['verts'][:1], inputs['faces'])
            elif k == 'lap' and weights[k] > 0:
                loss = laplacian_loss(outputs['verts'][:1], inputs['faces'])
            elif k == 'inflat' and weights[k] > 0:
                loss = inflation_loss(outputs['verts'][:1,:1], inputs['faces'])
                
            if k not in losses:
                losses[k] = []
            losses[k].append(loss * weights[k])

    def optimize(self, variables, inputs, params, weights, output_prefix=None):
        if 'pose_prior' in weights and not cfg.opt_instance:
            self.update_bone_rot_rest()

        losses = {}
        sigma = params['sigma'] if 'sigma' in params else 1e-4
        gamma = params['gamma'] if 'gamma' in params else 1e-4
        optimizer = torch.optim.Adam(variables, lr=params['lr'])
        loop = tqdm(range(params['n_iters']))        
        for j in loop:
            inputs['uvs'], inputs['faces'] = self.meshes[params['mesh_lev']].get_uvs_and_faces(gitter=True)
                
            # sigma decay
            if 'decay' in params and j%30 == 0 and j >= params['n_iters']//2:
                sigma = max([1e-7, sigma * params['decay']])
                gamma = max([1e-7, gamma * params['decay']])
                self.sil_renderer.set_sigma_gamma(sigma, gamma)
            
            # forward pass
            optimizer.zero_grad()
            if 'sil' in weights:
                params['stop_at'] = cfg.f_instance + j%5 if cfg.opt_instance else j%10
            else:
                params['stop_at'] = 10
            outputs = self.forward(inputs, deform=params['deform'], stop_at=params['stop_at'], text=params['text'])
            
            # calculate losses
            self.calculate_losses(inputs, outputs, losses, weights, params)
            loss = sum(losses[k][-1] for k in losses if len(losses[k]) > 0)
            loss.backward()
                
            # freeze certain parameters
            if 'pose_prior' in weights:
                self.freeze_parts(params['lev'])
                if cfg.opt_instance:
                    self.freeze_instances_except(cfg.instance_idx)
            
            # update parameters
            optimizer.step()
            if 'feat' in weights and not cfg.opt_instance:
                self.update_feat_verts(outputs['verts_2d'], inputs['feat_img'], inputs['masks_lr'])
        
        if output_prefix is not None:
            with torch.no_grad():
                self.save_results(inputs, params, losses, output_prefix)

    def train(self, inputs):
        # initialize vertex features
        self.feat_verts = inputs['feat_part'][self.part_mapping]
        
        print("========== Optimizing global pose... ========== ")
        var = [self.global_scale, self.global_trans, self.global_rot]
        params = {'n_iters':100, 'lr':0.05, 'lev':0, 'mesh_lev':2, 'deform':False, 'w_xy':2, 'text':None}
        weights = {'feat':0.5, 'part_cent':0.5, 'global_prior':0.5}
        self.optimize(var, inputs, params, weights, 'opt_global_')

        print("========== Optimizing pose 1... ========== ")
        var = [self.global_scale, self.global_trans, self.global_rot, self.bone_rot]
        params = {'n_iters':100, 'lr':0.05, 'lev':1, 'mesh_lev':2, 'deform':False, 'w_xy':2, 'text':None}
        weights = {'feat':1.0, 'global_prior':0.01, 'can_prior':0.1, 'pose_prior':0.01}
        self.optimize(var, inputs, params, weights)

        print("========== Optimizing pose 2... ========== ")
        params = {'n_iters':100, 'lr':0.05, 'lev':2, 'mesh_lev':2, 'deform':False, 'w_xy':2, 'text':None}
        outputs = self.optimize(var, inputs, params, weights)

        print("========== Optimizing pose 3... ========== ")
        params = {'n_iters':100, 'lr':0.05, 'lev':3, 'mesh_lev':2, 'deform':False, 'w_xy':2, 'text':None}
        self.optimize(var, inputs, params, weights, 'opt_pose_')
        
        print("========== Optimizing base shape... ========== ")
        var = [self.global_scale, self.global_trans, self.global_rot, self.bone_rot, self.part_codes]
        params = {'n_iters':100, 'lr':0.01, 'lev':1, 'mesh_lev':2, 'deform':False, 'w_xy':2, 'text':None}
        weights = {'feat':1.0, 'global_prior':0.01, 'can_prior':0.1, 'pose_prior':0.01, 'sil':0.05}
        self.optimize(var, inputs, params, weights)

        print("========== Optimizing base shape 2... ========== ")
        var = [self.global_scale, self.global_trans, self.global_rot, self.bone_rot, self.part_codes]
        params = {'n_iters':100, 'lr':0.01, 'lev':2, 'mesh_lev':2, 'deform':False, 'w_xy':2, 'text':None}
        self.optimize(var, inputs, params, weights)

        print("========== Optimizing base shape 3... ========== ")
        var = [self.global_scale, self.global_trans, self.global_rot, self.bone_rot, self.part_codes]
        params = {'n_iters':100, 'lr':0.01, 'lev':3, 'mesh_lev':2, 'deform':False, 'w_xy':2, 'text':None}
        self.optimize(var, inputs, params, weights, 'opt_base_')

        print("========== Optimizing shape... ========== ")
        var = [self.part_codes]
        for i, f in enumerate(self.f_parts):
            var += f.parameters()
        params = {'n_iters':100, 'lr':5e-3, 'lev':3, 'mesh_lev':2, 'deform':True, 'w_xy':1, 
                  'sigma':1e-4, 'gamma':1e-4, 'text':None}
        weights = {'feat':1.0, 'pose_prior':0.01, 'sil':0.1, 'shape_prior':0.1, 'lap':0.02, 'norm':0.02, 'inflat':1e-4}
        self.optimize(var, inputs, params, weights, 'opt_shape_')

        print("========== Optimizing all params... ========== ")
        var = [self.global_scale, self.global_trans, self.global_rot, self.bone_rot, self.part_codes]
        for i, f in enumerate(self.f_parts):
            var += f.parameters()
        params = {'n_iters':300, 'lr':5e-3, 'lev':3, 'mesh_lev':3, 'deform':True, 'w_xy':1, 
                  'sigma':1e-4, 'gamma':1e-4, 'decay':0.1, 'text':None}
        weights = {'feat':1.0, 'global_prior':0.01, 'can_prior':0.1, 'pose_prior':0.01,
                   'sil':0.1, 'shape_prior':0.1, 'lap':0.02, 'norm':0.02, 'inflat':1e-4}
        self.optimize(var, inputs, params, weights, 'opt_all_')

        print("========== Saving results... ========== ")
        with torch.no_grad():
            self.save_results(inputs, params)

    def optimize_instance(self, inputs):
        print("========== Optimizing %s instance %d shape... ========== " % (cfg.animal_class, cfg.instance_idx))
        var = []
        for i, f in enumerate(self.f_parts):
            var += f.parameters()
        params = {'n_iters':300, 'lr':1e-2, 'lev':3, 'mesh_lev':3, 'deform':True, 'sigma':1e-5, 'gamma':1e-5, 'text':None}
        weights = {'sil':0.05, 'instance':0.1, 'lap':0.1, 'norm':0.1}
        self.optimize(var, inputs, params, weights, 'opt_inst_')
        
    def save_results(self, inputs, params, losses=None, prefix=''):
        mesh_lev = 5 if cfg.opt_instance or prefix == '' else params['mesh_lev']
        uvs, faces = self.meshes[mesh_lev].get_uvs_and_faces()
        inputs['uvs'], inputs['faces'] = uvs, faces
        outputs = self.forward(inputs, stop_at=10, text='sample' if params['text'] is None else params['text'])
        verts_color = outputs['verts_color']
        
        num_imgs = inputs['images'].shape[0]
        if prefix == '' or cfg.opt_instance:
            for i in range(num_imgs):
                idx = cfg.instance_idx if cfg.opt_instance else i
                # 2D part masks
                masks = self.hard_renderer.get_part_masks(outputs['verts'][i:i+1], faces)
                img = inputs['images'][i].permute(1,2,0).cpu().numpy()
                cmask = part_mask_to_image(masks[0,0].cpu().numpy(), part_colors, img)
                save_img('%smask_pred_%d.png'%(prefix, idx), cmask)            
                # part and texture rendering
                img_part = self.hard_shaded_renderer.render(outputs['verts'][i:i+1], faces)
                img_text = self.hard_renderer.render(outputs['verts'][i:i+1], faces, verts_color[i:i+1])
                save_img('%spart_%d.png'%(prefix, idx), img2np(img_part))
                save_img('%stext_%d.png'%(prefix, idx), img2np(img_text))
                
        # part and texture gifs
        if cfg.opt_instance:
            for i in range(num_imgs):
                print('Rendering instance %d...' % i)
                idx = cfg.instance_idx if cfg.opt_instance else i
                imgs_part = []
                imgs_text = []
                for v in range(30):
                    verts = self.get_view(outputs['verts_can'][i:i+1], azimuth=v*12)
                    img_part = self.hard_shaded_renderer.render(verts, faces)
                    img_text = self.hard_renderer.render(verts, faces, verts_color[i:i+1])
                    imgs_part.append(img2np(img_part))
                    imgs_text.append(img2np(img_text))
                save_img('%spart_%d.gif'%(prefix, idx), imgs_part)
                save_img('%stext_%d.gif'%(prefix, idx), imgs_text)
            
        # animation
        if cfg.opt_instance:
            imgs = []
            global_scale, global_trans, global_rot, bone_rot = self.get_instance_params(idx=0)
            for v in range(16):
                w = v/7.5
                bone_rot2 = bone_rot * (1-w)
                rot = torch.cat([self.rot_id[None,None,:], bone_rot2 + self.bone_rot_rest[None,:,:]], 1)
                joints_can, joints_rot = self.skeleton.transform_joints(rot, scale=self.bone_scale)
                verts_can = self.transform_verts(inputs['uvs'], joints_can, joints_rot, True, 10)
                verts = self.global_transform(verts_can.reshape(1,-1,3), global_rot, global_trans, global_scale)
                img = self.hard_renderer.render(verts.reshape(1,cfg.nb,-1,3), faces, verts_color)
                imgs.append(img2np(img))
            imgs_rev = imgs[1:-1].copy()
            imgs_rev.reverse()
            save_img('%sanimation_%d.gif'%(prefix, cfg.instance_idx), imgs + imgs_rev)

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
        
    def load_model(self, model_path, load_parts=True, freeze_to=None):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.load_state_dict(checkpoint, strict=False)
        if load_parts and freeze_to is not None:
            for i, f in enumerate(self.f_parts):
                f.freeze_layers(freeze_to)
        
    def save_parts(self, model_path):
        torch.save(self.f_parts.state_dict(), model_path)
        
    def load_parts(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.f_parts.load_state_dict(checkpoint, strict=False)
        