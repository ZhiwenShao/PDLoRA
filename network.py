import torch
import torch.nn as nn
import os
import math
from hydra import initialize, compose
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
import loralib as lora

import torch.nn.functional as F
from dinov2.models.vision_transformer import vit_giant2

from timm.models.registry import register_model


class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

       
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
                      
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttHead(nn.Module):
    def __init__(self, channel= 512, width = 7):
        super().__init__()
        self.CoordAtt = CoordAtt(channel, channel, width)
    def forward(self, x):
        ca = self.CoordAtt(x)
        return ca  
        
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, width, groups=32):
        super(CoordAtt, self).__init__()
      
        self.Linear_h = Linear_block(inp, inp, groups=inp, kernel=(1, width), stride=(1, 1), padding=(0, 0))        
        self.Linear_w = Linear_block(inp, inp, groups=inp, kernel=(width, 1), stride=(1, 1), padding=(0, 0))
        
        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.Linear = Linear_block(oup, oup, groups=oup, kernel=(width, width), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten() 

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.Linear_h(x)
        x_w = self.Linear_w(x)
        x_w = x_w.permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        
        y = x_w * x_h
 
        return y


class DDAMNet(nn.Module):
    def __init__(self, au_num=12, num_head=2, channel= 512, width = 7):
        super(DDAMNet, self).__init__()

        self.au_num = au_num
        self.num_head = num_head
        self.channel = channel
        self.width = width
        for j in range(au_num):
            for i in range(num_head):
                setattr(self,"cat_head%d_%d" %(j,i), CoordAttHead(channel = channel, width = width))
        self.Linear = nn.ModuleList([
            Linear_block(channel, channel, groups=channel, kernel=(width, width), stride=(1, 1), padding=(0, 0))
            for j in range(au_num)])
        self.flatten = Flatten()      
        self.fc = nn.ModuleList([
            nn.Linear(channel, 1)
            for j in range(au_num)])
        
    def forward(self, x):

        for j in range(self.au_num):
            for i in range(self.num_head):

                tmp = getattr(self,"cat_head%d_%d" %(j,i))(x)
                tmp = tmp.unsqueeze(1)
                if i == 0:
                    heads = tmp
                else:
                    heads = torch.cat((heads, tmp), 1)
            head_out = heads.unsqueeze(1)

            y, _ = torch.max(heads, dim=1)
            
            y = x*y
            y = self.Linear[j](y)
            y = self.flatten(y) 
            out = self.fc[j](y)

            
            if j == 0:
                aus_output = out
                aus_attention = head_out
            else:
                aus_output = torch.cat((aus_output, out), 1)
                aus_attention = torch.cat((aus_attention, head_out), 1)

        return aus_attention, aus_output



def dinov2_lora(lora_depth = 2, r = 16, rand_rank=False, drop_rate=0.0):   
    
    dinov2 = vit_giant2(patch_size=14, img_size=526, init_values=1.0, ffn_layer='swiglufused', num_register_tokens=4, interpolate_antialias=True, interpolate_offset=0.0, block_chunks=0, lora_depth = lora_depth, r = r, rand_rank=rand_rank, drop_rate=drop_rate)
    dinov2.load_state_dict(torch.load('checkpoints/dinov2_vitg14_reg4_pretrain.pth'), strict=False) #dinov2 giant version
    lora.mark_only_lora_as_trainable(dinov2)

    return dinov2

      
def au_ddam(**kwargs):
    model = DDAMNet(**kwargs)
    return model



network_dict = {'dinov2_lora':dinov2_lora, 'au_ddam':au_ddam
}