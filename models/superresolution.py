import torch
from torch import nn
from eg3d.superresolution import SuperresolutionHybrid2X, SuperresolutionHybrid8XDC
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