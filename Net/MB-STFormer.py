
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
import einops



class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        self.conv = self.conv.to(x.device)
        x = self.conv(x)
        if self.bn:
            self.bn = self.bn.to(x.device)
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class LogPowerLayer(nn.Module):
    def __init__(self, dim):
        super(LogPowerLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(torch.mean(x ** 2, dim=self.dim), 1e-4, 1e4))



class InterFre(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        out = sum(x)
        out = F.gelu(out)
        return out
class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, doWeightNorm = True, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)



class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = self.weight.data.to(x.device)
            self.bias.data = self.bias.data.to(x.device)
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features=64, hidden_features=256, out_features=64, dropout=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()           # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, D) 这里 B=16, N=384, D=64
        x = self.fc1(x)               # (16, 384, 64) -> (16, 384, 256)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)               # (16, 384, 256) -> (16, 384, 64)
        x = self.drop(x)
        return x

class EfficientAdditiveAttnetion(nn.Module):


    def __init__(self, in_dims=64, token_dim=64, num_heads=1):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)
        norm_cfg = dict(type='BN2d', requires_grad=True)
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, 64))

        self.mlp = MLP(in_features=in_dims, hidden_features=int(in_dims * 4), out_features= in_dims, dropout=0.3)

    def forward(self, x):
        x = x.transpose(1 ,2)
        # x = torch.cat((self.cls_embedding.expand(x.shape[0], -1, -1), x), dim=1)
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1) # BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) # BxNxD

        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1

        A = torch.nn.functional.normalize(A, dim=1) # BxNx1

        G = torch.sum(A * query, dim=1) # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD

        out = self.Proj(G * key) + query # BxNxD

        out = self.final(out) # BxNxD


        out = out + x

        out = out + self.mlp(out)

        return out

class Stem(nn.Module):
    def __init__(self, in_planes, out_planes = 64,  patch_size = 192, radix = 5):
        nn.Module.__init__(self)
        norm_cfg = dict(type='BN2d', requires_grad=True)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        # self.kernel_size = kernel_size
        self.radix = radix
        self.patch_size = patch_size

        self.sconv = Conv(nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False, groups = radix),
                          bn=nn.BatchNorm1d(self.mid_planes), activation=None)

        self.tconv = nn.ModuleList()
        for i in range(self.radix):
            kernel_size = [125 ,63, 31 ,15 ,7]

            self.tconv.append(Conv
                (nn.Conv1d(self.out_planes, self.out_planes, kernel_size[i], 1, groups=self.out_planes, padding=kernel_size[i] // 2, bias=False ,),
                                   bn=nn.BatchNorm1d(self.out_planes), activation=None))


        self.interFre = InterFre()

        self.power = LogPowerLayer(dim=3)
        self.downSampling = nn.AvgPool1d(self.patch_size, self.patch_size)

        self.dp = nn.Dropout(0.5)
        self.effic = EfficientAdditiveAttnetion()
        self.pool = nn.AdaptiveAvgPool1d(12)


    def forward(self, x):
        N, C, T = x.shape
        out = self.sconv(x)

        out = torch.split(out, self.out_planes, dim=1)
        out = [m(x) for x, m in zip(out, self.tconv)]

        out = self.interFre(out)

        out = self.effic(out)
        out = out.permute(0 ,2 ,1)
        out = out.reshape(x.shape[0], 64, 2, 192)
        out = self.power(out)
        out = self.dp(out)

        return out


class MB_STFormer(nn.Module):
    def __init__(self, in_planes, out_planes, radix, patch_size, time_points, num_classes):

        nn.Module.__init__(self)
        self.in_planes = in_planes * radix
        self.out_planes = out_planes
        self.stem = Stem(self.in_planes, self.out_planes, patch_size=patch_size, radix=radix)

        self.fc1 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        out = self.stem(x)
        out = out.reshape(out.shape[0], -1)

        out = self.fc1(out)

        return out

