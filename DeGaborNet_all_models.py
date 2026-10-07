"""
@Time    : 2022/10/26 22:15
@Author  : Lin Luo
@FileName: regGabor2DNet.py
@describe TODO
"""
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torchsummary import summary
from torch.autograd import Function
from zdataset import applyPCA
from DeGaborNet_layer import *
from torch.cuda.amp import GradScaler, autocast

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))#定义权重矩阵
        if bias:#是否使用偏置项(bias)
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.weight, std=0.2)
        if self.bias is not None:
            init.constant_(self.bias, val=1e-2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        E = F.linear(input, self.weight, self.bias)
        return E


class DeGaborNet_fast_training_mode(nn.Module):
    def __init__(self, in_channels=5, num_classes=16, kernel=3, model_name='DeGaborNet', even_initial=True):
        """
        相位诱导的 Gabor 卷积核 网络，2块
        @param in_channels:
        @param num_classes:
        @param even_initial: whether adopt the even initialization strategy, false for random initialization
        """
        super(DeGaborNet_fast_training_mode, self).__init__()

        padding = kernel//2
        if model_name=='DeGaborNet':
            self.gfc = nn.Sequential(
                DeGCM_nojit(in_channels=in_channels, n_theta=4, n_omega=4, kernel_size=kernel, padding=padding,
                                 even_initial=even_initial, ty="Re"),
                DeGCM_nojit(in_channels=16, n_theta=8, n_omega=4, kernel_size=kernel, padding=padding,
                                  even_initial=even_initial, ty="Re"),
                DeGCM_nojit(in_channels=32, n_theta=16, n_omega=4, kernel_size=kernel, padding=padding,
                                  even_initial=even_initial, ty="Re"),
            )
        elif model_name=='DeGaborNet-R':
            self.gfc = nn.Sequential(
                DeGCM_nojit_R(in_channels=in_channels, n_theta=4, n_omega=4, kernel_size=kernel, padding=padding,
                                 even_initial=even_initial, ty="Re"),
                DeGCM_nojit_R(in_channels=16, n_theta=8, n_omega=4, kernel_size=kernel, padding=padding,
                                  even_initial=even_initial, ty="Re"),
                DeGCM_nojit_R(in_channels=32, n_theta=16, n_omega=4, kernel_size=kernel, padding=padding,
                                  even_initial=even_initial, ty="Re"),
            )
        self.cls = nn.Sequential(
            MyLinear(64, 128),
            nn.LeakyReLU(0.2),
            MyLinear(128, num_classes)
        )

    def forward(self, inputs, is_training_patch = True,patch_size = 15):
        out = self.gfc(inputs)
        if is_training_patch:
            out = torch.mean(out, dim=(2, 3))
            out=self.cls(out)
            return out
        else:
            # 推理：整图输入，已经外部padding好了
            # 直接做滑动窗口平均，不需要再padding
            out2 = F.avg_pool2d(
                out,
                kernel_size=patch_size,
                stride=1,
                padding=0  # 关键：不再padding！
            )  # [1, C, H_pad, W_pad] 尺寸不变

            # 后续处理...
            out = out2.permute(0, 2, 3, 1)  # [1, H_pad, W_pad, C]
            B, H_pad, W_pad, C = out.shape
            out = out.reshape(-1, C)

            out = self.cls(out)
            out = out.reshape(B, H_pad, W_pad, -1)
            return out

class DeGaborNet_fast_inference_mode(nn.Module):
    def __init__(self, in_channels=5, num_classes=16, kernel=3, model_name=0, even_initial=True):
        """
        @param in_channels:
        @param num_classes:
        @param even_initial: whether adopt the even initialization strategy, false for random initialization
        """
        super(DeGaborNet_fast_inference_mode, self).__init__()

        self.kernel_size = kernel
        padding = kernel//2
        if model_name==0:
            self.gfc = nn.Sequential(
                DeGCM_jit(in_channels=in_channels, n_theta=4, n_omega=4, kernel_size=kernel, padding=padding,
                                 even_initial=even_initial, ty=True),
                DeGCM_jit(in_channels=16, n_theta=8, n_omega=4, kernel_size=kernel, padding=padding,
                                  even_initial=even_initial, ty=True),
                DeGCM_jit(in_channels=32, n_theta=16, n_omega=4, kernel_size=kernel, padding=padding,
                                  even_initial=even_initial, ty=True),
            )
        else:
            self.gfc = nn.Sequential(
                DeGCM_jit_R(in_channels=in_channels, n_theta=4, n_omega=4, kernel_size=kernel, padding=padding,
                                 even_initial=even_initial, ty=True),
                DeGCM_jit_R(in_channels=16, n_theta=8, n_omega=4, kernel_size=kernel, padding=padding,
                                  even_initial=even_initial, ty=True),
                DeGCM_jit_R(in_channels=32, n_theta=16, n_omega=4, kernel_size=kernel, padding=padding,
                                  even_initial=even_initial, ty=True),
            )
        self.cls = nn.Sequential(
            MyLinear(64, 128),
            nn.LeakyReLU(0.2),
            MyLinear(128, num_classes)
        )

    def forward(self,inputs: torch.Tensor,is_training_patch: bool = True,patch_size: int = 15) -> torch.Tensor:
        out = self.gfc(inputs)
        if is_training_patch:
            out = torch.mean(out, dim=(2, 3))
            out=self.cls(out)
            return out
        else:
            # 推理：整图输入，已经外部padding好了
            # 直接做滑动窗口平均，不需要再padding
            out2 = F.avg_pool2d(
                out,
                kernel_size=patch_size,
                stride=1,
                padding=0  # 关键：不再padding！
            )  # [1, C, H_pad, W_pad] 尺寸不变

            # 后续处理...
            out = out2.permute(0, 2, 3, 1)  # [1, H_pad, W_pad, C]
            B, H_pad, W_pad, C = out.shape
            out = out.reshape(-1, C)

            out = self.cls(out)
            out = out.reshape(B, H_pad, W_pad, -1)
            return out
