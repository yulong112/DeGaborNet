import time
import torch
import torch.nn.functional as F
from torch import nn
# from chainer import cuda
from DeGaborNet_ops import *
# import cupy as cp
from torch.cuda.amp import GradScaler, autocast

# cudnn = cuda.cuda.cudnn

class DeGCM_nojit(nn.Module):
    def __init__(self, in_channels, n_theta, n_omega, kernel_size, strides=1, padding=0, even_initial=True,
                 use_bias=True, requires_grad=True, ty="Re"):
        """
        @param in_channels: img band number
        @param n_theta: number of theta initializations (int)
        @param n_omega: number of omegas initializations (int)
        @param kernel_size: 5
        @param strides: 1
        @param padding: 2
        @param even_initial: whether adopt the even initialization strategy, false for random initialization
        @param use_bias:
        @param requires_grad:
        """
        super(DeGCM_nojit, self).__init__()
        self.requires_grad = requires_grad
        self.even_initial = even_initial
        self.padding = padding
        self.strides = strides
        self.kernel_size = kernel_size
        self.n_omega = n_omega
        self.n_theta = n_theta
        self.ichannel = in_channels
        self.ochannel = n_omega * n_theta
        self.ty = ty
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(n_theta * n_omega))
        else:
            self.bias = None
        n_out = n_theta * n_omega  # 为该卷积层输出的通道数
        self.theta = nn.Parameter(get_theta(1, n_theta, n_omega, even_initial=True),requires_grad=requires_grad)
        self.omega = nn.Parameter(get_omega(1, n_theta, n_omega, even_initial=True),requires_grad=requires_grad)
        self.sigma = nn.Parameter(get_sigma(1, n_out, kernel_size=kernel_size, even_initial=True),requires_grad=requires_grad)
        self.phase_x = nn.Parameter(get_phase(self.ichannel, n_out, even_initial=False),requires_grad=requires_grad)
        self.phase_y = nn.Parameter(get_phase(self.ochannel, n_out, even_initial=False),requires_grad=requires_grad)
        # self.phase_x1 = nn.Parameter(get_phase(self.ichannel,1, even_initial=False), requires_grad=requires_grad)
        # self.phase_y1 = nn.Parameter(get_phase(self.ochannel, 1, even_initial=False), requires_grad=requires_grad)

        # origin version
        self.relu1 = nn.ELU(2)
        self.bn1 = nn.BatchNorm2d(num_features=self.ochannel)
        self.relu2 = nn.ELU(2)
        self.bn2 = nn.BatchNorm2d(num_features=self.ochannel)
        # self.relu2 = nn.ReLU()
        # # self.relu2 = nn.LeakyReLU(0.2)
        # self.bn2 = nn.BatchNorm2d(num_features=self.ochannel)

        # new version
        # self.bn0 = nn.BatchNorm2d(num_features=self.ichannel)
        # self.relu10 = nn.ReLU()
        # self.bn2_0 = nn.BatchNorm2d(num_features=self.ochannel)
        # self.relu20 = nn.ReLU()
        # self.bn2 = nn.BatchNorm2d(num_features=self.ochannel)
        # self.bn2_1 = nn.BatchNorm2d(num_features=self.ochannel)

        # self.phase_y.requires_grad = False
        # self.phase_x.requires_grad = False
    def forward(self, inputs):
        # new version
        # inputs = self.bn0(inputs)
        # W_x = get_1DGabor_kernels_x(theta=self.theta, omega=self.omega, sigma=self.sigma, phase=self.phase_x,
        #                             kernel_size=self.kernel_size, in_channels=self.ichannel,out_channels=self.ochannel, device=inputs.device,
        #                             g_type=self.ty)
        # inputs1 = F.pad(inputs, (self.padding, self.padding, 0, 0))  # (left, right, top, bottom)
        # out1 = F.conv2d(inputs1, W_x, self.bias, stride=self.strides)
        # out1 = self.relu10(out1)
        # out1 = self.bn2_0(out1)
        # W_y = get_1DGabor_kernels_y(theta=self.theta, omega=self.omega, sigma=self.sigma, phase=self.phase_y,
        #                             kernel_size=self.kernel_size, in_channels=self.ochannel,out_channels=self.ochannel, device=inputs.device,
        #                             g_type=self.ty)
        # out2 = F.pad(out1, (0, 0, self.padding, self.padding))
        # E1 = F.conv2d(out2, W_y, self.bias, stride=self.strides)
        # E1 = self.bn2(E1)
        # E1 = self.relu20(E1)
        # E1 = self.bn2_1(E1)

        # origin version
        # inputs = self.bn0(inputs)
        W_x = get_1DGabor_kernels_x(theta=self.theta, omega=self.omega, sigma=self.sigma, phase=self.phase_x,
                                    kernel_size=self.kernel_size, in_channels=self.ichannel,out_channels=self.ochannel, device=inputs.device,
                                    g_type=self.ty)
        inputs1 = F.pad(inputs, (self.padding, self.padding, 0, 0))  # (left, right, top, bottom)
        out1 = F.conv2d(inputs1, W_x, self.bias, stride=self.strides)
        out1 = self.relu1(out1)
        out1 = self.bn1(out1)
        W_y = get_1DGabor_kernels_y(theta=self.theta, omega=self.omega, sigma=self.sigma, phase=self.phase_y,
                                    kernel_size=self.kernel_size, in_channels=self.ochannel,out_channels=self.ochannel, device=inputs.device,
                                    g_type=self.ty)
        out2 = F.pad(out1, (0, 0, self.padding, self.padding))
        E1 = F.conv2d(out2, W_y, self.bias, stride=self.strides)
        E1 = self.relu2(E1)
        E1 = self.bn2(E1)

        return E1

class DeGCM_nojit_R(nn.Module):
    def __init__(self, in_channels, n_theta, n_omega, kernel_size, strides=1, padding=0, even_initial=True,
                 use_bias=True, requires_grad=True, ty="Re"):
        """
        @param in_channels: img band number
        @param n_theta: number of theta initializations (int)
        @param n_omega: number of omegas initializations (int)
        @param kernel_size: 5
        @param strides: 1
        @param padding: 2
        @param even_initial: whether adopt the even initialization strategy, false for random initialization
        @param use_bias:
        @param requires_grad:
        """
        super(DeGCM_nojit_R, self).__init__()
        self.requires_grad = requires_grad
        self.even_initial = even_initial
        self.padding = padding
        self.strides = strides
        self.kernel_size = kernel_size
        self.n_omega = n_omega
        self.n_theta = n_theta
        self.ichannel = in_channels
        self.ochannel = n_omega * n_theta
        self.ty = ty
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(n_theta * n_omega))
        else:
            self.bias = None
        n_out = n_theta * n_omega  # 为该卷积层输出的通道数
        self.sigma = nn.Parameter(get_sigma(1, n_out, kernel_size=kernel_size, even_initial=True),requires_grad=requires_grad)
        self.theta = nn.Parameter(get_theta(1, n_theta, n_omega, even_initial=True),requires_grad=requires_grad)
        self.omega = nn.Parameter(get_omega(1, n_theta, n_omega, even_initial=True),requires_grad=requires_grad)
        self.phase_x = nn.Parameter(get_phase(self.ichannel, n_out, even_initial=False),requires_grad=requires_grad)
        self.phase_y = nn.Parameter(get_phase(self.ochannel, n_out, even_initial=False),requires_grad=requires_grad)

        # origin version
        self.relu1 = nn.ELU(2)
        self.bn1 = nn.BatchNorm2d(num_features=self.ochannel)
        self.relu2 = nn.ELU(2)
        self.bn2 = nn.BatchNorm2d(num_features=self.ochannel)

        self.fusion = nn.Sequential(
            nn.Conv2d(self.ochannel*2, self.ochannel, kernel_size=1),
            nn.ReLU()
        )
    def forward(self, inputs):
        W_x = get_1DGabor_kernels_x(theta=self.theta, omega=self.omega, sigma=self.sigma, phase=self.phase_x,
                                    kernel_size=self.kernel_size, in_channels=self.ichannel,out_channels=self.ochannel, device=inputs.device,
                                    g_type=self.ty)
        inputs1 = F.pad(inputs, (self.padding, self.padding, 0, 0))  # (left, right, top, bottom)
        out1 = F.conv2d(inputs1, W_x, self.bias, stride=self.strides)
        out1 = self.relu1(out1)
        out1 = self.bn1(out1)
        W_y = get_1DGabor_kernels_y_ab_scattering_others(theta=self.theta, omega=self.omega, sigma=self.sigma, phase=self.phase_y,
                                    kernel_size=self.kernel_size, in_channels=self.ochannel,out_channels=self.ochannel, device=inputs.device,
                                    g_type=self.ty)                                                                                                  # ablation: no scattering
        W_y2 = get_1DGabor_kernels_y(theta=self.theta, omega=self.omega, sigma=self.sigma, phase=self.phase_y,
                                    kernel_size=self.kernel_size, in_channels=self.ochannel, out_channels=self.ochannel, device=inputs.device,
                                    g_type=self.ty)
        out2 = F.pad(out1, (0, 0, self.padding, self.padding))
        E1 = F.conv2d(out2, W_y, self.bias, stride=self.strides)
        E1 = self.relu2(E1)
        E1 = self.bn2(E1)
        E2 = F.conv2d(out2, W_y2, self.bias, stride=self.strides)
        E2 = self.relu2(E2)
        E2 = self.bn2(E2)

        # 1. 拼接
        concat = torch.cat([E1, E2], dim=1)  # [B, 32, H, W]
        E1 = self.fusion(concat)  # [B, 16, H, W]

        return E1


class DeGCM_jit_R(nn.Module):
    def __init__(self, in_channels, n_theta, n_omega, kernel_size, strides=1, padding=0, even_initial=True,
                 use_bias=True, requires_grad=True, ty=True):
        super(DeGCM_jit_R, self).__init__()
        self.requires_grad = requires_grad
        self.even_initial = even_initial
        self.padding = padding
        self.strides = strides
        self.kernel_size = kernel_size
        self.n_omega = n_omega
        self.n_theta = n_theta
        self.ichannel = in_channels
        self.ochannel = n_omega * n_theta
        self.ty = ty
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(n_theta * n_omega))
        else:
            self.bias = None
        n_out = n_theta * n_omega  # 为该卷积层输出的通道数
        self.theta = nn.Parameter(get_theta(1, n_theta, n_omega, even_initial=True),requires_grad=requires_grad)
        self.omega = nn.Parameter(get_omega(1, n_theta, n_omega, even_initial=True),requires_grad=requires_grad)
        self.sigma = nn.Parameter(get_sigma(1, n_out, kernel_size=kernel_size, even_initial=True),requires_grad=requires_grad)
        self.phase_x = nn.Parameter(get_phase(self.ichannel, n_out, even_initial=False),requires_grad=requires_grad)
        self.phase_y = nn.Parameter(get_phase(self.ochannel, n_out, even_initial=False),requires_grad=requires_grad)  # ablation: no scattering

        # origin version
        self.relu1 = nn.ELU(2.0)
        self.bn1 = nn.BatchNorm2d(num_features=self.ochannel)
        self.relu2 = nn.ELU(2.0)
        self.bn2 = nn.BatchNorm2d(num_features=self.ochannel)

        self.fusion = nn.Sequential(
            nn.Conv2d(self.ochannel*2, self.ochannel, kernel_size=1),
            nn.ReLU()
        )
        # self.forward = torch.compile(self.forward, mode="max-autotune")
    def forward(self, inputs):

        W_x = get_DeGCMkernels_x(theta=self.theta, omega=self.omega, sigma=self.sigma, phase=self.phase_x,
                                    kernel_size=self.kernel_size, in_channels=self.ichannel,out_channels=self.ochannel, device=inputs.device,
                                    g_type=self.ty)
        inputs1 = F.pad(inputs, (self.padding, self.padding, 0, 0))  # (left, right, top, bottom)
        out1 = F.conv2d(inputs1, W_x, self.bias, stride=self.strides)
        out1 = self.relu1(out1)
        out1 = self.bn1(out1)

        W_y = get_DeGCMkernels_y_ab_scattering_others(theta=self.theta, omega=self.omega, sigma=self.sigma, phase=self.phase_y,
                                    kernel_size=self.kernel_size, in_channels=self.ochannel,out_channels=self.ochannel, device=inputs.device,
                                    g_type=self.ty)
        W_y2 = get_DeGCMkernels_y(theta=self.theta, omega=self.omega, sigma=self.sigma, phase=self.phase_y,
                                    kernel_size=self.kernel_size, in_channels=self.ochannel,out_channels=self.ochannel, device=inputs.device,
                                    g_type=self.ty)
        out2 = F.pad(out1, (0, 0, self.padding, self.padding))
        E1 = F.conv2d(out2, W_y, self.bias, stride=self.strides)
        E1 = self.relu2(E1)
        E1 = self.bn2(E1)
        E2 = F.conv2d(out2, W_y2, self.bias, stride=self.strides)
        E2 = self.relu2(E2)
        E2 = self.bn2(E2)

        # 1. 拼接
        concat = torch.cat([E1, E2], dim=1)  # [B, 32, H, W]
        E1 = self.fusion(concat)  # [B, 16, H, W]
        return E1

class DeGCM_jit(nn.Module):
    def __init__(self, in_channels, n_theta, n_omega, kernel_size, strides=1, padding=0, even_initial=True,
                 use_bias=True, requires_grad=True, ty=True):
        super(DeGCM_jit, self).__init__()
        self.requires_grad = requires_grad
        self.even_initial = even_initial
        self.padding = padding
        self.strides = strides
        self.kernel_size = kernel_size
        self.n_omega = n_omega
        self.n_theta = n_theta
        self.ichannel = in_channels
        self.ochannel = n_omega * n_theta
        self.ty = ty
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(n_theta * n_omega))
        else:
            self.bias = None
        n_out = n_theta * n_omega  # 为该卷积层输出的通道数
        self.theta = nn.Parameter(get_theta(1, n_theta, n_omega, even_initial=True),requires_grad=requires_grad)
        self.omega = nn.Parameter(get_omega(1, n_theta, n_omega, even_initial=True),requires_grad=requires_grad)
        self.sigma = nn.Parameter(get_sigma(1, n_out, kernel_size=kernel_size, even_initial=True),requires_grad=requires_grad)
        self.phase_x = nn.Parameter(get_phase(self.ichannel, n_out, even_initial=False),requires_grad=requires_grad)
        self.phase_y = nn.Parameter(get_phase(self.ochannel, n_out, even_initial=False),requires_grad=requires_grad)

        # origin version
        self.relu1 = nn.ELU(2.0)
        self.bn1 = nn.BatchNorm2d(num_features=self.ochannel)
        self.relu2 = nn.ELU(2.0)
        self.bn2 = nn.BatchNorm2d(num_features=self.ochannel)

        # self.forward = torch.compile(self.forward, mode="max-autotune")
    def forward(self, inputs):

        W_x = get_DeGCMkernels_x(theta=self.theta, omega=self.omega, sigma=self.sigma, phase=self.phase_x,
                                    kernel_size=self.kernel_size, in_channels=self.ichannel,out_channels=self.ochannel, device=inputs.device,
                                    g_type=self.ty)
        W_y = get_DeGCMkernels_y(theta=self.theta, omega=self.omega, sigma=self.sigma, phase=self.phase_y,
                                    kernel_size=self.kernel_size, in_channels=self.ochannel,out_channels=self.ochannel, device=inputs.device,
                                    g_type=self.ty)

        inputs1 = F.pad(inputs, (self.padding, self.padding, 0, 0))  # (left, right, top, bottom)
        out1 = F.conv2d(inputs1, W_x, self.bias, stride=self.strides)
        out1 = self.relu1(out1)
        out1 = self.bn1(out1)
        out2 = F.pad(out1, (0, 0, self.padding, self.padding))
        E1 = F.conv2d(out2, W_y, self.bias, stride=self.strides)
        E1 = self.relu2(E1)
        E1 = self.bn2(E1)

        return E1
