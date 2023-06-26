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
import json
import cv2
import numpy as np
import collections
from scipy import ndimage
from skimage.morphology import thin
from argparse import ArgumentParser
from config import cfg
from data_utils import *
from dataloader import *    
    
    
def get_max_connected_component(masks):
    bs, hw = masks.shape[0], masks.shape[-1]
    for i in range(bs):
        labels, n_feat = ndimage.label(masks[i,0])
        max_component = 1
        n_pixels = np.sum(labels == 1)
        for c in range(2, n_feat+1):
            if np.sum(labels == c) > n_pixels:
                n_pixels = np.sum(labels == c)
                max_component = c
        masks[i,0] *= labels == max_component
    return masks


def bfs(grid, start, end):
    height, width = grid.shape[:2]
    queue = collections.deque([[start]])
    seen = set([start])
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if x == end[0] and y == end[1]:
            return np.array(path)
        for x2, y2 in [(x+1,y), (x-1,y), (x,y+1), (x,y-1), (x+1,y+1), (x-1,y-1), (x-1,y+1), (x+1,y-1)]:
            if 0 <= x2 < width and 0 <= y2 < height and grid[y2,x2] > 0 and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))
    return None


def get_2d_skeleton(outputs, mask):
    hw = mask.shape[-1]    
    # distance transform and thinning
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)      
    thinned = thin((mask>0).astype(np.uint8)).astype(np.float32)
    y, x = np.where(thinned > 0)
    indices = np.argsort(dist[y,x])[::-1]
    x, y = x[indices], y[indices]
    
    # junction and end points
    kernel = np.ones((3,3), dtype=np.float32)
    filtered = cv2.filter2D(thinned, -1, kernel).astype(np.float32)
    junction_map = (thinned > 0) * (filtered > 3)
    endpoint_map = (thinned > 0) * (filtered == 2)   
    y_junc, x_junc = np.where(junction_map)
    y_end, x_end = np.where(endpoint_map)    
    indices = np.argsort(dist[y_junc,x_junc])[::-1]
    x_junc, y_junc = x_junc[indices], y_junc[indices]
    
    # label skeleton points
    for idx in range(x.shape[0]):
        thinned[y[idx], x[idx]] = idx+1
    thinned = thinned.astype(np.int32)
    
    pts = np.stack([x, y, dist[y,x]], 1)
    junctions = (thinned[y_junc, x_junc]-1).astype(np.int32)
    endpoints = (thinned[y_end, x_end]-1).astype(np.int32)
    
    # BFS
    paths = []
    root = tuple(pts[junctions[0],:2].astype(np.int32))
    for p in junctions[1:]:
        target = tuple(pts[p,:2].astype(np.int32))
        paths.append(bfs(thinned, root, target))
    for p in endpoints:
        target = tuple(pts[p,:2].astype(np.int32))
        paths.append(bfs(thinned, root, target))
        
    # 2D skeleton graph    
    junctions_filtered = []
    endpoints_filtered = []
    ancestors = np.zeros(pts.shape[0]).astype(np.int32)
    for path in paths:
        ancestor = junctions[0]
        for x, y in path:
            pt_idx = thinned[y,x] - 1
            ancestors[pt_idx] = ancestor
            if pt_idx in junctions:
                ancestor = pt_idx
                junctions_filtered.append(pt_idx)
            elif pt_idx in endpoints:
                endpoints_filtered.append(pt_idx)    
    junctions = np.array(junctions_filtered)
    endpoints = np.array(endpoints_filtered)
    
    # filter junction and endpoints
    thres_dist_endpoints = 1.0
    thres_dist_junctions = 0.8
    for i1 in range(junctions.shape[0]):
        for i2 in range(endpoints.shape[0]):
            p1, p2 = junctions[i1].copy(), endpoints[i2].copy()
            if p2 >= 0 and np.sum((pts[p1,:2]-pts[p2,:2])**2) < (pts[p1,2]*thres_dist_endpoints)**2:
                ancestors[p2] = 0
                endpoints[i2] = -1
    for i1 in range(junctions.shape[0]):
        for i2 in range(i1+1, junctions.shape[0]):
            p1, p2 = junctions[i1].copy(), junctions[i2].copy()
            if p2 >= 0 and np.sum((pts[p1,:2]-pts[p2,:2])**2) < (pts[p1,2]*thres_dist_junctions)**2:
                junctions[i2] = -1
                ancestors = np.where(ancestors == p2, p1, ancestors)                
    endpoints = np.stack([j for j in endpoints if j>=0], 0)
    junctions = np.stack([j for j in junctions if j>=0], 0)

    # final joints
    joint_indices = []
    for j in junctions:
        if j not in joint_indices:
            joint_indices.append(j)
    for j in endpoints:
        if j not in joint_indices:
            joint_indices.append(j)
    joint_indices = np.stack(joint_indices, 0)    
    joints_3d = np.stack([np.zeros_like(pts[joint_indices,0]), pts[joint_indices,1], pts[joint_indices,0]], 1)
    joints_3d = (joints_3d - joints_3d[:1])/hw # root centered and normalized
    joints_3d[:,1] *= -1
    
    joints_parent = np.stack([np.where(joint_indices == p)[0][0] for p in ancestors[joint_indices]])
    joint_indices = joint_indices[1:]
    part_dino = mask[pts[joint_indices,1].astype(np.int32), pts[joint_indices,0].astype(np.int32)] - 1
    part_length = np.sqrt(np.sum((pts[joint_indices,:2] - pts[ancestors[joint_indices],:2])**2, 1))
    part_scale_xz = np.minimum(pts[joint_indices,2], pts[ancestors[joint_indices],2]) / part_length
    part_scale_y = 1 + np.minimum(pts[joint_indices,2], pts[ancestors[joint_indices],2])*2 / part_length
      
    outputs['nb'] = joints_3d.shape[0]-1
    outputs['joints'] = joints_3d
    outputs['joints_parent'] = joints_parent
    outputs['part_dino'] = part_dino
    outputs['part_scale_xz'] = part_scale_xz
    outputs['part_scale_y'] = part_scale_y   

    
def find_symmetric_pairs(outputs):
    for i1 in range(1, outputs['joints'].shape[0]):
        j1 = outputs['joints'][i1]
        p1 = outputs['joints_parent'][i1]
        f1 = outputs['part_dino'][i1-1]
        d1 = (j1[1] - outputs['joints'][p1][1])**2 + (j1[2] - outputs['joints'][p1][2])**2
        if j1[0] != 0:
            continue
        for i2 in range(i1+1, outputs['joints'].shape[0]):
            j2 = outputs['joints'][i2]
            p2 = outputs['joints_parent'][i2]
            f2 = outputs['part_dino'][i2-1]
            d2 = (j2[1] - outputs['joints'][p2][1])**2 + (j2[2] - outputs['joints'][p2][2])**2
            if j2[0] != 0:
                continue
            if p1 == p2 and f1 == f2 and d1/d2 < 2 and d1/d2 > 0.5:
                return [i1, i2]
    return []
            
def lift_skeleton_to_3d(outputs):
    symmetry_pairs = find_symmetric_pairs(outputs)
    while len(symmetry_pairs) > 0:
        j1, j2 = symmetry_pairs
        outputs['joints'][j1,0] = -0.05
        outputs['joints'][j2,0] = 0.05
        joint_mean = (outputs['joints'][j1] + outputs['joints'][j2])/2
        for c, p in enumerate(outputs['joints_parent']):
            if p == j1:
                outputs['joints'][c,1:] += joint_mean[1:] - outputs['joints'][j1,1:]
                outputs['joints'][c,0] = outputs['joints'][j1,0]
            elif p == j2:
                outputs['joints'][c,1:] += joint_mean[1:] - outputs['joints'][j2,1:]
                v[c,0] = outputs['joints'][j2,0]
        outputs['joints'][j1,1:], outputs['joints'][j2,1:] = joint_mean[1:], joint_mean[1:]
        part_scale_xz_mean = (outputs['part_scale_xz'][j1-1] + outputs['part_scale_xz'][j2-1])/2
        part_scale_y_mean = (outputs['part_scale_y'][j1-1] + outputs['part_scale_y'][j2-1])/2
        outputs['part_scale_xz'][j1-1], outputs['part_scale_xz'][j2-1] = part_scale_xz_mean, part_scale_xz_mean
        outputs['part_scale_y'][j1-1], outputs['part_scale_y'][j2-1] = part_scale_y_mean, part_scale_y_mean

        parent = outputs['joints_parent'][j1]
        if parent != 0 and sum([p == parent for p in outputs['joints_parent']]) == 2:
            outputs['nb'] += 1
            outputs['joints_parent'] = np.array([p+1 if p > parent else p for p in outputs['joints_parent']])
            outputs['joints_parent'][j2] += 1
            outputs['joints_parent'] = np.insert(outputs['joints_parent'], parent, outputs['joints_parent'][parent])
            outputs['joints'] = np.concatenate((outputs['joints'][:parent+1], outputs['joints'][parent:]))
            outputs['joints'][parent,0] = -0.05
            outputs['joints'][parent+1,0] = 0.05
            outputs['part_scale_xz'][parent-1] *= 0.5
            outputs['part_scale_xz'] = np.concatenate((outputs['part_scale_xz'][:parent], outputs['part_scale_xz'][parent-1:]))
            outputs['part_scale_y'] = np.concatenate((outputs['part_scale_y'][:parent], outputs['part_scale_y'][parent-1:]))
            outputs['part_dino'] = np.concatenate((outputs['part_dino'][:parent], outputs['part_dino'][parent-1:]))
            
        symmetry_pairs = find_symmetric_pairs(outputs)
    
    
def extract_skeleton(masks):   
    bs, hw = masks.shape[0], masks.shape[-1]
    masks = get_max_connected_component(masks)    
    
    img_xy = np.arange(hw).astype(np.float32)
    grid_y, grid_x = np.meshgrid(img_xy, img_xy, indexing='ij')
    cluster_present = np.ones((cfg.n_clusters))
    cluster_xy = np.zeros((bs,cfg.n_clusters,2))
    for i in range(bs):
        cent_x = np.mean(grid_x[masks[i,0]>0])
        cent_y = np.mean(grid_y[masks[i,0]>0])
        for j in range(cfg.n_clusters):
            m = masks[i,0] == j+1
            if np.sum(m) == 0:
                cluster_present[j] = 0
            else:
                part_cent_x = np.mean(grid_x[m])
                part_cent_y = np.mean(grid_y[m])
                cluster_xy[i,j,0] = (part_cent_x - cent_x)
                cluster_xy[i,j,1] = (part_cent_y - cent_y)
                
    cluster_x_mean = np.mean(np.abs(cluster_xy[:,:,0]), axis=0)
    anchor_cluster = np.argmax(cluster_x_mean * cluster_present)
    # anchor_cluster = np.argmax(cluster_x_mean)
    anchor_is_head = np.mean(cluster_xy[:,anchor_cluster,1]) < 0
    facing_right = np.ones(bs).astype(np.float32)
    for i in range(bs):
        flip = anchor_is_head and cluster_xy[i,anchor_cluster,0] < 0
        flip = flip or (not anchor_is_head and cluster_xy[i,anchor_cluster,0] > 0)
        if flip:
            facing_right[i] = -1
            cluster_xy[i,:,0] *= -1
            masks[i,0,:,:] = masks[i,0,:,::-1]

    indices = np.argsort(np.mean(np.abs(cluster_xy[:,:,0]), axis=1))
    azim = np.arange(bs).astype(np.float32)/(bs-1)
    azim *= np.pi/2
    azim = np.stack([facing_right[i] * azim[np.where(indices==i)[0][0]] for i in range(bs)])
    outputs = {}
    outputs['azimuth'] = azim.tolist()
    
    get_2d_skeleton(outputs, masks[cfg.instance_idx,0])
    lift_skeleton_to_3d(outputs)
    
    side_view = np.where(indices == cfg.instance_idx)[0][0] >= 5
    if not side_view:
        outputs['joints'][:,0], outputs['joints'][:,2] = outputs['joints'][:,2].copy(), outputs['joints'][:,0].copy()
        
    outputs['joints'] = outputs['joints'].tolist()
    outputs['joints_parent'] = outputs['joints_parent'].tolist()
    outputs['part_dino'] = outputs['part_dino'].tolist()
    outputs['part_scale_xz'] = outputs['part_scale_xz'].tolist()
    outputs['part_scale_y'] = outputs['part_scale_y'].tolist()
    json.dump(outputs, open(osp.join(cfg.model_dir, 'skeleton_%s.json' % cfg.animal_class), 'w'))    
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cls', type=str, default='zebra', dest='cls')
    parser.add_argument('--inst', type=bool, default=False, dest='opt_instance')
    parser.add_argument('--idx', type=int, default=0, dest='instance_idx')
    args = parser.parse_args()
    cfg.set_args(args)
        
    num_imgs, inputs = load_data()
    masks = inputs['masks'].cpu().numpy().astype(np.uint8)
    extract_skeleton(masks)
    print("Finished extracting skeleton.")
    