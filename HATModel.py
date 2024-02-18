import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import sys
import math


class Swin_cpas_offset(nn.Module):
    def __init__(self, num_channels=3, train_enc=True, load_weight=1):
        super(Swin_cpas_offset, self).__init__()
        self.backbone = models.swin_b(weights=torchvision.models.Swin_B_Weights.IMAGENET1K_V1).features
        self.index_len = 3
        pos_dim = self.index_len * 4

        for param in self.backbone.parameters():
            param.requires_grad = train_enc

        self.Fuse3 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.Fuse2 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.Fuse1 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.Fuse0 = nn.Sequential(
            nn.Conv2d(8+pos_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.Offset = nn.Sequential(
            nn.Conv2d(pos_dim, pos_dim, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(pos_dim, pos_dim, kernel_size=1, padding=0),
            nn.ReLU(),
        )

    def get_coords(self, shape):
        # Creates the position encoding
        coords_type = 'cosine'
        bs, _, h, w = shape
        if coords_type == 'cosine':
            for i in range(0, self.index_len):
                x = torch.arange(0, w).float()
                y = torch.arange(0, h).float()
                xcos = torch.cos((x.float() / (10000 ** (2*i/2))).float())
                xsin = torch.sin((x.float() / (10000 ** (2*i/2))).float())
                ycos = torch.cos((y.float() / (10000 ** (2*i/2))).float())
                ysin = torch.sin((y.float() / (10000 ** (2*i/2))).float())
                xcos = xcos.view(1, 1, 1, w).repeat(bs, 1, h, 1)
                xsin = xsin.view(1, 1, 1, w).repeat(bs, 1, h, 1)
                ycos = ycos.view(1, 1, h, 1).repeat(bs, 1, 1, w)
                ysin = ysin.view(1, 1, h, 1).repeat(bs, 1, 1, w)
                coords_cur = torch.cat([xcos, xsin, ycos, ysin], 1)
                if not i == 0:
                    coords = torch.cat([coords, coords_cur], 1)
                else:
                    coords = coords_cur
        else:
            raise NotImplementedError()
        # coords = coords.view(coords.shape[0], coords.shape[1], -1).permute(0, 2, 1)
        return coords.cuda()

    def forward(self, images):
        B, C, H, W = images.shape

        imgs = images
        x = []
        for index, block in enumerate(self.backbone):
            imgs = block(imgs)
            if index % 2 == 1:
                x.append(imgs)
        # x = [r(_x.permute(0, 3, 1, 2)) for r, _x in zip(self.reduce_channels, x)]
        for index, imgs in enumerate(x):
            imgs = imgs.permute(0, 3, 1, 2)
            scale = H // imgs.shape[2]
            imgs = nn.PixelShuffle(upscale_factor=scale)(imgs)
            x[index] = imgs

        # pred block
        coords = self.get_coords(shape=x[3].shape)
        inp = x[3]
        pred3 = self.Fuse3(inp)

        coords = self.get_coords(shape=x[2].shape)
        inp = pred3+x[2]
        pred2 = self.Fuse2(inp)

        coords = self.get_coords(shape=x[1].shape)
        inp = pred2+x[1]
        pred1 = self.Fuse1(inp)

        coords = self.get_coords(shape=x[0].shape)
        coords = self.Offset(coords)
        inp = pred1+x[0]
        inp = torch.cat([coords, inp], 1)
        pred0 = self.Fuse0(inp)

        return pred0
