import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from utils import *
from dataLoader.ray_utils import ndc_rays_blender
from models.render_modules import positional_encoding
import torch.nn.functional as F
from super_image import EdsrModel


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):
    rgbs, depth_maps = [], []
    N_rays_all = rays.shape[0]

    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
    
        rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
    
    return torch.cat(rgbs), torch.cat(depth_maps)


def OctreeRender_trilinear_fast_SR(rays, cam_params, tensorf, img_wh=(64,64), chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):
    rgb_maps, depth_maps = [], []
    N_rays_all = rays.shape[0]
    W, H = img_wh

    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
    
        rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)

    rgb_maps = torch.cat(rgb_maps)
    rgbs = tensorf.sr_module(cam_params, rgb_maps, (W, H))
    
    return rgbs, torch.cat(depth_maps)