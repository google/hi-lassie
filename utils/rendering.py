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
from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    look_at_rotation,
    look_at_view_transform,
    PerspectiveCameras,
    OrthographicCameras,
    AmbientLights,
    PointLights,
    BlendParams,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    SoftPhongShader,
    HardPhongShader,
    Textures,
    TexturesVertex,
)
from config import cfg
from data_utils import *
    
    
class Renderer():
    def __init__(self, device, shader, light):
        super().__init__()
        self.device = device
        self.shader = shader
        R, T = look_at_view_transform(dist=5, elev=0, azim=0, device=device)
        self.cams = OrthographicCameras(device=device, focal_length=1, R=R, T=T)
        self.part_color = torch.zeros(cfg.nb,3).float().to(device)
        for i in range(cfg.nb):
            self.part_color[i,:] = torch.tensor(part_colors[i+1][:3]).to(device)

        # Light settings
        if light == 'ambient':
            self.lights = AmbientLights(device=device)
            
        elif light == 'point':
            self.lights = PointLights(
                device=device, location=[[0.0, 1.0, 2.0]], ambient_color=((0.5,0.5,0.5),),
                diffuse_color=((0.3,0.3,0.3),), specular_color=((0.2,0.2,0.2),))
        
        # Shader settings
        if shader == 'soft_sil':
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0,0.0,0.0)) 
            raster_settings = RasterizationSettings(
                image_size=cfg.input_size,
                blur_radius=np.log(1./1e-4-1.)*blend_params.sigma,
                faces_per_pixel=50,
                bin_size=0,
            )
            shader = SoftSilhouetteShader(blend_params=blend_params)
            
        elif shader == 'soft_phong':
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0,0.0,0.0)) 
            raster_settings = RasterizationSettings(
                image_size=cfg.input_size,
                blur_radius=np.log(1./1e-4-1.)*blend_params.sigma*0.5,
                faces_per_pixel=50,
                bin_size=0,
            )
            shader = SoftPhongShader(device=device, cameras=self.cams[0], lights=self.lights, blend_params=blend_params)
            
        else:
            raster_settings = RasterizationSettings(
                image_size=cfg.input_size,
                blur_radius=0.0,
                faces_per_pixel=1,
                bin_size=0,
                max_faces_per_bin=100,
            )
            shader = HardPhongShader(device=device, cameras=self.cams[0], lights=self.lights)
            
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cams[0], raster_settings=raster_settings),
            shader=shader
        ).to(device)
    
    def render(self, verts, faces, verts_rgb=None, part_idx=-1):
        bs = verts.shape[0]
        if len(verts.shape) == 3:
            nv = verts.shape[1]
            verts_combined = verts
            faces_combined = faces[None,:,:].repeat(bs,1,1)
        else:
            nb, nv = verts.shape[1], verts.shape[2]
            verts_combined = verts.permute(0,3,1,2).reshape(bs,3,-1).permute(0,2,1)
            faces_combined = torch.cat([faces + i*nv for i in range(nb)], 0)[None,:,:].repeat(bs,1,1)

        if self.shader == 'soft_sil':
            mesh = Meshes(verts=verts_combined, faces=faces_combined)
        else:
            if verts_rgb is None:
                if part_idx == -1:
                    verts_rgb = self.part_color[None,:,None,:].repeat(bs,1,nv,1)
                else:
                    verts_rgb = self.part_color[None,part_idx,None,:].repeat(bs,1,nv,1)
            verts_rgb = verts_rgb.permute(0,3,1,2).reshape(bs,3,-1).permute(0,2,1)
            mesh = Meshes(verts=verts_combined, faces=faces_combined, textures=TexturesVertex(verts_rgb))            
        return self.renderer(mesh, cameras=self.cams[0])

    def project(self, x):
        return self.cams[0].transform_points_screen(x, image_size=cfg.input_size)
    
    def set_sigma_gamma(self, sigma, gamma):
        blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=(0.0,0.0,0.0)) 
        self.renderer.shader.blend_params = blend_params
        self.renderer.rasterizer.raster_settings.blur_radius = np.log(1./1e-4-1.)*sigma*0.5
    
    def get_part_masks(self, verts, faces):
        bs, nb, nv = verts.shape[:3]
        masks = []
        for i in range(bs):
            verts_combined = verts[i:i+1].permute(0,3,1,2).reshape(1,3,-1).permute(0,2,1)
            faces_combined = torch.cat([faces + i*nv for i in range(nb)], 0)[None]
            meshes = Meshes(verts=verts_combined, faces=faces_combined)
            fragments = self.renderer.rasterizer(meshes, cameras=self.cams[0])
            mask = torch.div(fragments.pix_to_face, faces.shape[0], rounding_mode='floor')+1  # (1, H, W, 1)
            mask[fragments.pix_to_face == -1] = 0
            masks.append(mask)
        return torch.cat(masks, 0).permute(0,3,1,2)
    
    def get_verts_vis(self, verts, faces):
        bs, nb, nv = verts.shape[:3]
        verts_vis = []
        for i in range(bs):
            verts_combined = verts[i:i+1].permute(0,3,1,2).reshape(1,3,-1).permute(0,2,1)
            faces_combined = torch.cat([faces + i*nv for i in range(nb)], 0)[None]
            meshes = Meshes(verts=verts_combined, faces=faces_combined)
            packed_faces = meshes.faces_packed() 
            pix_to_face = self.renderer.rasterizer(meshes, cameras=self.cams[0]).pix_to_face # (1, H, W, 1)
            visible_faces = pix_to_face.unique()
            visible_verts = torch.unique(packed_faces[visible_faces])
            visibility_map = torch.zeros_like(verts_combined[0,:,0])
            visibility_map[visible_verts] = 1
            verts_vis.append(visibility_map.view(nb, nv))
        return torch.stack(verts_vis, 0)
