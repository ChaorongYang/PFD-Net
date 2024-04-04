import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import copy
import numpy as np
from collections import OrderedDict
from mmcv.ops import DeformConv2dPack as DCN

import math
#pvt 侧面输出结合FFC代码
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Spectra(nn.Module):
    def __init__(self, in_depth, AF='prelu'):
        super().__init__()

        # Params
        self.in_depth = in_depth
        self.inter_depth = self.in_depth // 2 if in_depth >= 2 else self.in_depth

        # Layers
        self.AF1 = nn.ReLU if AF == 'relu' else nn.PReLU(self.inter_depth)
        self.AF2 = nn.ReLU if AF == 'relu' else nn.PReLU(self.inter_depth)
        self.inConv = nn.Sequential(nn.Conv2d(self.in_depth, self.inter_depth, 1),
                                    nn.BatchNorm2d(self.inter_depth),
                                    self.AF1)
        self.midConv = nn.Sequential(nn.Conv2d(self.inter_depth, self.inter_depth, 1),
                                     nn.BatchNorm2d(self.inter_depth),
                                     self.AF2)
        self.outConv = nn.Conv2d(self.inter_depth, self.in_depth, 1)

    def forward(self, x):
        x = self.inConv(x)
        _, _, H, W = x.shape
        skip = copy.copy(x)
        rfft = torch.fft.rfft2(x)
        real_rfft = torch.real(rfft)
        imag_rfft = torch.imag(rfft)
        cat_rfft = torch.cat((real_rfft, imag_rfft), dim=-1)
        cat_rfft = self.midConv(cat_rfft)
        mid = cat_rfft.shape[-1] // 2
        real_rfft = cat_rfft[..., :mid]
        imag_rfft = cat_rfft[..., mid:]
        rfft = torch.complex(real_rfft, imag_rfft)
        spect = torch.fft.irfft2(rfft,s=(H, W))
        out = self.outConv(spect + skip)
        return out


class FastFC(nn.Module):
    def __init__(self, in_depth, AF='prelu'):
        super().__init__()
        # Params
        self.in_depth = in_depth // 2

        # Layers
        self.AF1 = nn.ReLU if AF == 'relu' else nn.PReLU(self.in_depth)
        self.AF2 = nn.ReLU if AF == 'relu' else nn.PReLU(self.in_depth)
        self.conv_ll = nn.Conv2d(self.in_depth, self.in_depth, 3,padding='same')
        self.conv_lg = nn.Conv2d(self.in_depth, self.in_depth, 3,padding='same')
        self.conv_gl = nn.Conv2d(self.in_depth, self.in_depth, 3,padding='same')
        self.conv_gg = Spectra(self.in_depth, AF)
        self.bnaf1 = nn.Sequential(nn.BatchNorm2d(self.in_depth), self.AF1)
        self.bnaf2 = nn.Sequential(nn.BatchNorm2d(self.in_depth), self.AF2)

    def forward(self, x):
        mid = x.shape[1] // 2
        x_loc = x[:, :mid, :, :]
        x_glo = x[:, mid:, :, :]
        x_ll = self.conv_ll(x_loc)
        x_lg = self.conv_lg(x_loc)
        x_gl = self.conv_gl(x_glo)
        x_gg = self.conv_gg(x_glo)
        # print('x_ll', x_ll.size())
        # print('x_lg', x_lg.size())
        # print('x_gl', x_gl.size())
        # print('x_gg', x_gg.size())

        out_loc = torch.add((self.bnaf1(x_ll + x_gl)), x_loc)
        out_glo = torch.add((self.bnaf2(x_gg + x_lg)), x_glo)
        out = torch.cat((out_loc, out_glo), dim=1)
        return out, out_loc, out_glo


class FourierBlock(nn.Module):
    def __init__(self, num_layer, in_depth, return_all=True):
        super().__init__()
        # Params
        self.num_layers = num_layer
        self.in_depth = in_depth
        self.return_all = return_all
        # Layers
        self.block = nn.ModuleList()
        for _ in range(self.num_layers):
            self.block.append(FastFC(self.in_depth, 'prelu'))

    def forward(self, x):
        x0=x
        for layer in self.block:
            x, x_loc, x_glo = layer(x)
        if self.return_all:
            return x, x_loc, x_glo
        else:
            return x+x0


class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()

        self.c1 = Conv2D(channel, channel, kernel_size=3, padding=1)


        self.s3 = Conv2D(channel, channel, kernel_size=1, padding=0, act=False)

        self.relu = nn.ReLU(inplace=True)

        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=5)

        self.deform = DCN(channel // 4, channel // 4, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = ConvBNR(channel, channel, 3)
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        s = x
        x1 = self.c1(x)

        xc = torch.chunk(x1, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.deform(xc[3] + x)
        x = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))


        s = self.s3(s)
        x = self.relu(x + s)

        x = self.conv3_3(x)
        x = x * self.ca(x)
        x = x * self.sa(x)

        return x

class multiconv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(2*in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(2*in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(2*in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel,  kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(2*in_channel, out_channel, 1),
        )
        self.deform = DCN(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.branch4_1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channel),
        )
        self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(2*in_channel, out_channel, 1)


        self.ca = ChannelAttention(out_channel)
        self.sa = SpatialAttention()

    def forward(self, x1,x2):
        t = x1
        x = torch.cat([x1, x2], axis=1)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x4 = self.branch4_1(self.deform(self.branch4(x)))

        x_cat = self.conv_cat(torch.cat((x0, x1, x2,x4), 1))

        x = self.relu(x_cat + self.conv_res(x))
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x+t

class residual_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.network = nn.Sequential(
            Conv2D(in_c, out_c),
            Conv2D(out_c, out_c, kernel_size=1, padding=0, act=False)

        )
        self.shortcut = Conv2D(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_init):
        x = self.network(x_init)
        s = self.shortcut(x_init)
        x = self.relu(x+s)
        return x

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = residual_block(in_c[0]+in_c[1], out_c)
        self.r2 = residual_block(out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        x = self.r2(x)
        return x




class PFD_Net(nn.Module):
    def __init__(self, n_classes=1):
        super(PFD_Net, self).__init__()
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        
        #CNN
        self.inc = ConvBatchNorm(3, 16)
        self.Down1 = DownBlock(16, 16 * 2, nb_Conv=2)
        self.Down2 = DownBlock(16 * 2, 16 * 4, nb_Conv=2)

        
        self.FB1 = FourierBlock(2, 64, False)
        self.FB2 = FourierBlock(2,128, False)
        self.FB3 = FourierBlock(2, 320, False)
        self.FB4 = FourierBlock(2, 512, False)

        self.c2   = nn.Conv2d(128, 64, 1, bias=False)
        self.c3   = nn.Conv2d(320, 64, 1, bias=False)
        self.c4   = nn.Conv2d(512, 64, 1, bias=False)

        self.q1 = CAM(64)
        self.q2 = CAM(64)
        self.q3 = CAM(64)
        self.q4 = CAM(64)

        self.d1 = decoder_block([64, 64], 64)
        self.d2 = decoder_block([64, 64], 64)
        self.d3 = decoder_block([64, 64], 64)

        self.w1 = multiconv(64,64)

        self.out1 = nn.Conv2d(64, n_classes, 1)
        self.out2 = nn.Conv2d(64, n_classes, 1)
        self.out3 = nn.Conv2d(64, n_classes, 1)
        self.out4 = nn.Conv2d(64, n_classes, 1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        # backbone
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)


        t1=self.Down2(self.Down1(self.inc(x)))
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        f1 = self.FB1(x1)
        f2 = self.FB2(x2)
        f3 = self.FB3(x3)
        f4 = self.FB4(x4)

        #通道数目减少
        f2 = self.c2(f2)
        f3 = self.c3(f3)
        f4 = self.c4(f4)

        f1 = self.q1(f1)
        f2 = self.q2(f2)
        f3 = self.q3(f3)
        f4 = self.q4(f4)

        d1 = self.d1(f4, f3)
        d2 = self.d2(d1, f2)
        d3 = self.d3(d2, f1
                     
        pre= self.w1(d3,t1)
                     
        pre = self.out1(pre)
        pre1= self.out2(d3)
        pre2= self.out3(d2)
        pre3= self.out4(d1)
        p1 = F.interpolate(pre, scale_factor=4, mode='bilinear')
        p2 = F.interpolate(pre1, scale_factor=4, mode='bilinear')
        p3 = F.interpolate(pre2, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(pre3, scale_factor=16, mode='bilinear')


        return p1,p2,p3,p4