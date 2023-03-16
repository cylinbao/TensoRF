import torch
from torch import nn
from eg3d.superresolution import SuperresolutionHybrid2X, SuperresolutionHybrid4X, SuperresolutionHybrid8XDC
from eg3d.networks_stylegan2 import MappingNetwork
from super_image import EdsrModel, EdsrConfig


class Interpolation(torch.nn.Module):
    def __init__(self, sr_ratio=1):
        super(Interpolation, self).__init__()
        self.scale_ratio = sr_ratio

    # def forward(self, rgb, x, ws, **block_kwargs):
    def forward(self, rgb):
        sr_rgb = nn.functional.interpolate(rgb, scale_factor=self.scale_ratio, mode='bilinear', align_corners=False, antialias=True)

        return sr_rgb 


class SR_Module(torch.nn.Module):
    def __init__(self, sr_method, device, sr_ratio=1.0):
        super(SR_Module, self).__init__()
        self.sr_method = sr_method
        self.sr_ratio = sr_ratio
        self.channels = 32

        if self.sr_ratio > 1:
            if sr_method == "Eg3d":
                self.mapping = MappingNetwork(z_dim=0, c_dim=25, w_dim=512, num_ws=14, num_layers=2).to(device)
                if self.sr_ratio == 2.0:
                    sr_args = {'channels': 32, 'img_resolution': 128, 'sr_num_fp16_res': 4, 'sr_antialias': True, 
                               'channel_base': 32768, 'channel_max': 48, 'fused_modconv_default': 'inference_only'}
                    self.sr = SuperresolutionHybrid2X(**sr_args).to(device)
                elif self.sr_ratio == 4.0:
                    sr_args = {'channels': 32, 'img_resolution': 256, 'sr_num_fp16_res': 4, 'sr_antialias': True, 
                               'channel_base': 32768, 'channel_max': 48, 'fused_modconv_default': 'inference_only'}
                    self.sr = SuperresolutionHybrid4X(**sr_args).to(device)
                elif self.sr_ratio == 8.0:
                    sr_args = {'channels': 32, 'img_resolution': 512, 'sr_num_fp16_res': 4, 'sr_antialias': True, 
                               'channel_base': 32768, 'channel_max': 48, 'fused_modconv_default': 'inference_only'}
                    self.sr = SuperresolutionHybrid8XDC(**sr_args).to(device)
                else:
                    raise NotImplementedError(f"ratio {sr_ratio}x for {sr_method} is not supported.")
            elif sr_method == "Bilinear":
                self.sr = Interpolation(sr_ratio=self.sr_ratio).to(device)
            elif sr_method == "Edsr":
                edsr_config = EdsrConfig(scale=self.sr_ratio)
                self.sr = EdsrModel(edsr_config).to(device)
                # self.sr = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=self.scale_ratio)
            else:
                raise NotImplementedError(f"{sr_method} is not implemented.")


    def forward(self, cam_params, feat_maps, img_wh):
        W, H = img_wh

        if self.sr_ratio == 1:
            rgbs = feat_maps[:,:3]
            sr_image = rgbs
        else:
            rgbs = feat_maps[:,:3].permute(1, 0).view(1, -1, W, H)

            if self.sr_method == "Eg3d":
                feat_maps = feat_maps.permute(1, 0).view(1, -1, W, H)
                ws = self.mapping(None, cam_params, truncation_psi=0.7, truncation_cutoff=14)
                sr_image = self.sr(rgbs, feat_maps, ws)
            else:
                sr_image = self.sr(rgbs)

            sr_image = sr_image.permute(0,2,3,1).view(-1,3)

        return sr_image 