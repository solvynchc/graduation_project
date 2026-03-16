import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        assert style in ['lp', 'pl']
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            assert in_channels % (scale ** 2) == 0
            in_channels = in_channels // (scale ** 2)

        self.scale = scale
        self.style = style
        self.groups = groups
        self.dyscope = dyscope
        self.in_channels = in_channels

        out_channels = 2 * groups * (scale ** 2 if style == 'lp' else 1)
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.constant_(self.offset.bias, 0)

        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            nn.init.constant_(self.scope.weight, 0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        mesh = torch.meshgrid(h, h, indexing='ij')
        pos = torch.stack(mesh).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)
        return pos

    def sample(self, x, offset):
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)

        coords_h = torch.arange(h, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
        yy, xx = torch.meshgrid(coords_h, coords_w, indexing='ij')
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).unsqueeze(2)

        normalizer = torch.tensor([w, h], device=x.device, dtype=x.dtype).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = F.pixel_shuffle(coords.view(b, -1, h, w), self.scale)
        coords = coords.view(b, 2, -1, self.scale * h, self.scale * w)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        x = x.reshape(b * self.groups, -1, h, w)
        x = F.grid_sample(
            x,
            coords,
            mode='bilinear',
            align_corners=False,
            padding_mode='border'
        )
        return x.view(b, -1, self.scale * h, self.scale * w)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)
