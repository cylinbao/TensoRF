import torch
from torch import nn
from eg3d.superresolution import SuperresolutionHybrid2X, SuperresolutionHybrid8XDC
from eg3d.networks_stylegan2 import MappingNetwork
from models.render_modules import positional_encoding

class SuperResolution(torch.nn.Module):
    def __init__(self):
        super(SuperResolution, self).__init__()

        self.conv = nn.Conv2d(3, 32, kernel_size=1, stride=1)
        self.mapping = nn.Sequential(self.conv, torch.nn.ReLU(), nn.BatchNorm2d(32))

        # sr_args = {'channels': 32, 'img_resolution': 128, 'sr_num_fp16_res': 4, 'sr_antialias': True, 
        #            'channel_base': 32768, 'channel_max': 48, 'fused_modconv_default': 'inference_only'}
        # self.sr = SuperresolutionHybrid2X(**sr_args).cuda()

        sr_args = {'channels': 32, 'img_resolution': 512, 'sr_num_fp16_res': 4, 'sr_antialias': True, 
                   'channel_base': 32768, 'channel_max': 48, 'fused_modconv_default': 'inference_only'}
        self.sr = SuperresolutionHybrid8XDC(**sr_args).cuda()

        self.init_weight()
    
    def init_weight(self):
        nn.init.kaiming_uniform_(self.conv.weight.data, nonlinearity='relu')
        nn.init.constant_(self.conv.bias.data, 0)

    def forward(self, rays, rgbs):
        ws = positional_encoding(rays[0,:3,0,0], 12).view(1,1,-1)
        feat_map = self.mapping(rgbs)
        rgb_map = feat_map[:, :3]
        sr_image = self.sr(rgb_map, feat_map, ws)

        return sr_image


class Interpolation(torch.nn.Module):
    def __init__(self, sr_ratio=1):
        super(Interpolation, self).__init__()
        self.scale_ratio = sr_ratio

    def forward(self, rgb, x, ws, **block_kwargs):
        sr_rgb = nn.functional.interpolate(rgb, scale_factor=self.scale_ratio, mode='bilinear', align_corners=False, antialias=True)

        return sr_rgb 

class SR_Module(torch.nn.Module):
    def __init__(self, device, sr_ratio=1):
        super(SR_Module, self).__init__()
        self.scale_ratio = sr_ratio
        self.channels = 32
        self.mapping = MappingNetwork(z_dim=0, c_dim=25, w_dim=512, num_ws=14, num_layers=2).to(device)

        # selecting sr module
        if self.scale_ratio == 1:
            self.sr = torch.nn.Sequential(
                torch.nn.Linear(self.channels, 3),
                torch.nn.Sigmoid()
            ).to(device)
            torch.nn.init.constant_(self.sr[0].bias, 0)
        else:
            # sr_module = Interpolation(sr_ratio=args.sr_ratio).to(device)

            sr_args = {'channels': 32, 'img_resolution': 128, 'sr_num_fp16_res': 4, 'sr_antialias': True, 
                       'channel_base': 32768, 'channel_max': 48, 'fused_modconv_default': 'inference_only'}
            self.sr = SuperresolutionHybrid2X(**sr_args).to(device)

            # sr_args = {'channels': 32, 'img_resolution': 512, 'sr_num_fp16_res': 4, 'sr_antialias': True, 
            #            'channel_base': 32768, 'channel_max': 48, 'fused_modconv_default': 'inference_only'}
            # sr_module = SuperresolutionHybrid8XDC(**sr_args).to(device)


    def forward(self, cam_params, rgbs, feat_maps, img_wh, device):
        W, H = img_wh
        rgbs = rgbs.permute(1, 0).view(1, -1, W, H)
        feat_maps = feat_maps.permute(1, 0).view(1, -1, W, H)

        if self.scale_ratio == 1:
            sr_image = self.sr(feat_maps)
        else:
            ws = self.mapping(None, cam_params, truncation_psi=0.7, truncation_cutoff=14)
            sr_image = self.sr(rgbs, feat_maps, ws)
            sr_image = sr_image.permute(0,2,3,1).view(-1,3)

        return sr_image 