import torch
from eg3d.superresolution import SuperresolutionHybrid2X, SuperresolutionHybrid8X

class SuperResolution(torch.nn.Module):
    def __init__(self):
        super(SuperResolution, self).__init__()
        self.sr2x = SuperresolutionHybrid2X(channels=32, img_resolution=128, sr_num_fp16_res=1, sr_antialias=True).cuda()
        self.sr8x = SuperresolutionHybrid8X(channels=32, img_resolution=512, sr_num_fp16_res=1, sr_antialias=True).cuda()

    def forward(self):
      rgb = torch.randn([1,3,64,64]).cuda()
      feat2x = torch.randn([1,32,64,64]).cuda()
      ws2x = torch.randn([1,14,512]).cuda()
      out = self.sr2x(rgb, feat2x, ws2x)

      feat8x = torch.randn([1,32,128,128]).cuda()
      ws8x = torch.randn([1,14,512]).cuda()
      out = self.sr8x(out, feat8x, ws8x)

      return