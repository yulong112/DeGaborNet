"""
Detailed Gabor-Nets Implementation
"""

import numpy as np
import torch
import math
from math import pi
from typing import Union


# 获得一维卷积核w

def get_1DGabor_kernels_x1(theta: torch.Tensor, omega: torch.Tensor, sigma: torch.Tensor, phase: torch.Tensor,
                           kernel_size: int, device='cuda:0', g_type='Re'):
    """
    Create 1D Gabor kernels

    :param theta:
    :param omega:
    :param sigma:
    :param phase:
    :param kernel_size: 卷积核大小
    :param g_type:
    :param device
    :param requires_grad: whether to use gradients (boolean)

    :return:
    """
    sigma = sigma.unsqueeze(-1).to(device)  # [out, in, 1]
    phase = phase.unsqueeze(-1).to(device)  # [out, in, 1]

    # 计算频率分量
    Freq_X = (omega * torch.cos(theta)).unsqueeze(-1).to(device)  # [out, in, 1]

    # 创建欧几里得坐标
    coords = L2_grid(kernel_size)[np.newaxis, np.newaxis, :]  # (1, 1, kernel_size)
    coords_X = torch.tensor(coords, dtype=torch.float32, device=device)

    # 计算包络
    Envelop = torch.exp(-0.5 * (coords_X ** 2) / (sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)  # (out, in, kernel_size)

    # 构建Gabor核
    Freq = Freq_X * coords_X  # (out, in, kernel_size)
    if g_type == 'Re':
        kernels = Envelop * torch.cos(Freq + phase)
    elif g_type == 'Im':
        kernels = Envelop * torch.sin(Freq + phase)
    else:
        raise ValueError('Error Gabor filter type')

    return kernels.unsqueeze(2)  # 返回结果


def get_1DGabor_kernels_y1(theta: torch.Tensor, omega: torch.Tensor, sigma: torch.Tensor, phase: torch.Tensor,
                           kernel_size: int, device='cuda:0', g_type='Im'):
    """
       Create 1D Gabor kernels

       :param theta:
       :param omega:
       :param sigma:
       :param phase:
       :param kernel_size: 卷积核大小
       :param g_type:
       :param device
       :param requires_grad: whether to use gradients (boolean)

       :return:
       """
    sigma = sigma.unsqueeze(-1)  # [out, in, 1]
    phase = phase.unsqueeze(-1)  # [out, in, 1]

    # Freq_X = omega * torch.cos(theta)  # [out, in]
    Freq_Y = omega * torch.sin(theta)  # [out, in]
    Freq_Y = Freq_Y.unsqueeze(-1)  # [out, in, 1]

    # Create the Euclidean coordinates of the kernel
    coords = L2_grid(kernel_size)[np.newaxis, np.newaxis, :]  # (1, 1, kernel_size)
    coords_Y = torch.tensor(coords, dtype=torch.float32, device=device)

    # Envelop construction
    Envelop = torch.exp(-0.5 * (coords_Y ** 2) / (sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)  # (out, in, kernel_size)

    # Construct corresponding Gabor kernels
    Freq = Freq_Y * coords_Y  # (out, in, kernel_size)

    if g_type == 'Re':
        kernels = Envelop * torch.cos(Freq + phase)
    elif g_type == 'Im':
        kernels = Envelop * torch.sin(Freq + phase)
    else:
        raise ValueError('Error Gabor filter type')

    return kernels.unsqueeze(2).permute(0, 1, 3, 2)


def get_1DGabor_kernels_x(theta: torch.Tensor, omega: torch.Tensor, sigma: torch.Tensor, phase: torch.Tensor,
                          kernel_size: int, in_channels=103, out_channels=16, device='cuda:0', g_type='Re'):
    # sh = list(theta.shape)              # [out,in]
    # 在索引为-1上增加一个维度，二维 -> 三维
    sigma = sigma.expand(-1, in_channels)
    # phase = phase.expand(out_channels, -1)
    omega = omega.expand(-1, in_channels)
    theta = theta.expand(-1, in_channels)

    sigma = torch.unsqueeze(sigma, -1)  # [out,in, 1]
    phase = torch.unsqueeze(phase, -1)  # [out,in, 1]

    Freq_X = (omega * torch.cos(theta)).unsqueeze(-1).to(device)  # [out, in, 1]

    # 创建欧几里得坐标
    coords = L2_grid(kernel_size)[np.newaxis, np.newaxis, :]  # (1, 1, kernel_size)
    coords_X = torch.tensor(coords, dtype=torch.float32, device=device)

    # 计算包络
    Envelop = torch.exp(-0.5 * (coords_X ** 2) / (sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)  # (out, in, kernel_size)

    # 构建Gabor核
    Freq = Freq_X * coords_X  # (out, in, kernel_size)
    if g_type == 'Re':
        kernels = Envelop * torch.cos(Freq + phase)
    elif g_type == 'Im':
        kernels = Envelop * torch.sin(Freq + phase)
    else:
        raise ValueError('Error Gabor filter type')

    return kernels.unsqueeze(2)  # 返回结果


def get_1DGabor_kernels_y(theta: torch.Tensor, omega: torch.Tensor, sigma: torch.Tensor, phase: torch.Tensor,
                          kernel_size: int, in_channels=103, out_channels=16, device='cuda:0', g_type='Im'):
    sigma = sigma.expand(-1, in_channels)
    # phase = phase.expand(out_channels, -1)
    omega = omega.expand(-1, in_channels)
    theta = theta.expand(-1, in_channels)
    # sigma = sigma.T                #ablation: no scattering
    # theta = theta.T                #ablation: no scattering
    sigma = sigma.unsqueeze(-1)  # [out, in, 1]
    phase = phase.unsqueeze(-1)  # [out, in, 1]

    # Freq_X = omega * torch.cos(theta)  # [out, in]
    Freq_Y = omega * torch.sin(theta)  # [out, in]
    Freq_Y = Freq_Y.unsqueeze(-1)  # [out, in, 1]

    # Create the Euclidean coordinates of the kernel
    coords = L2_grid(kernel_size)[np.newaxis, np.newaxis, :]  # (1, 1, kernel_size)
    coords_Y = torch.tensor(coords, dtype=torch.float32, device=device)

    # Envelop construction
    Envelop = torch.exp(-0.5 * (coords_Y ** 2) / (sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)  # (out, in, kernel_size)

    # Construct corresponding Gabor kernels
    Freq = Freq_Y * coords_Y  # (out, in, kernel_size)

    if g_type == 'Re':
        kernels = Envelop * torch.cos(Freq + phase)
    elif g_type == 'Im':
        kernels = Envelop * torch.sin(Freq + phase)
    else:
        raise ValueError('Error Gabor filter type')

    return kernels.unsqueeze(2).permute(0, 1, 3, 2)


def get_1DGabor_kernels_x_ab_channelshare(theta: torch.Tensor, omega: torch.Tensor, sigma: torch.Tensor, phase: torch.Tensor,
                          kernel_size: int, in_channels=103, out_channels=16, device='cuda:0', g_type='Re'):
    # sh = list(theta.shape)              # [out,in]
    # 在索引为-1上增加一个维度，二维 -> 三维
    # sigma = sigma.expand(-1, in_channels)  #ablation: channel sharing
    # # phase = phase.expand(out_channels, -1)
    # omega = omega.expand(-1, in_channels)
    # theta = theta.expand(-1, in_channels)
    omega = omega[:,:in_channels]  #ablation: no channel sharing
    theta = theta[:,:in_channels]  #ablation: no channel sharing

    sigma = torch.unsqueeze(sigma, -1)  # [out,in, 1]
    phase = torch.unsqueeze(phase, -1)  # [out,in, 1]

    Freq_X = (omega * torch.cos(theta)).unsqueeze(-1).to(device)  # [out, in, 1]

    # 创建欧几里得坐标
    coords = L2_grid(kernel_size)[np.newaxis, np.newaxis, :]  # (1, 1, kernel_size)
    coords_X = torch.tensor(coords, dtype=torch.float32, device=device)

    # 标记 sigma 为 0 的位置
    zero_mask = (sigma.data.abs() < 1e-10)  # 找出所有为 0 的 sigma
    # 将 sigma 为 0 的位置临时设为 1（避免除零）
    sigma.data[zero_mask] = 1.0

    # 计算包络
    Envelop = torch.exp(-0.5 * (coords_X ** 2) / (sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)  # (out, in, kernel_size)

    # 🔑 关键：将 sigma 为 0 对应的核全部置为 0
    if zero_mask.any():
        Envelop = Envelop * (~zero_mask).float()

    # 构建Gabor核
    Freq = Freq_X * coords_X  # (out, in, kernel_size)
    if g_type == 'Re':
        kernels = Envelop * torch.cos(Freq + phase)
    elif g_type == 'Im':
        kernels = Envelop * torch.sin(Freq + phase)
    else:
        raise ValueError('Error Gabor filter type')

    return kernels.unsqueeze(2)  # 返回结果


def get_1DGabor_kernels_y_ab_channelshare(theta: torch.Tensor, omega: torch.Tensor, sigma: torch.Tensor, phase: torch.Tensor,
                          kernel_size: int, in_channels=103, out_channels=16, device='cuda:0', g_type='Im'):
    # sigma = sigma.expand(-1, in_channels)  #ablation: channel sharing
    # # phase = phase.expand(out_channels, -1)
    # omega = omega.expand(-1, in_channels)
    # theta = theta.expand(-1, in_channels)
    omega = omega[:,:in_channels]  #ablation: no channel sharing
    theta = theta[:,:in_channels]  #ablation: no channel sharing
    # sigma = sigma.T                #ablation: no scattering
    # theta = theta.T                #ablation: no scattering
    sigma = sigma.unsqueeze(-1)  # [out, in, 1]
    phase = phase.unsqueeze(-1)  # [out, in, 1]

    # Freq_X = omega * torch.cos(theta)  # [out, in]
    Freq_Y = omega * torch.sin(theta)  # [out, in]
    Freq_Y = Freq_Y.unsqueeze(-1)  # [out, in, 1]

    # Create the Euclidean coordinates of the kernel
    coords = L2_grid(kernel_size)[np.newaxis, np.newaxis, :]  # (1, 1, kernel_size)
    coords_Y = torch.tensor(coords, dtype=torch.float32, device=device)

    # 标记 sigma 为 0 的位置
    zero_mask = (sigma.data.abs() < 1e-10)  # 找出所有为 0 的 sigma
    # 将 sigma 为 0 的位置临时设为 1（避免除零）
    sigma.data[zero_mask] = 1.0

    # Envelop construction
    Envelop = torch.exp(-0.5 * (coords_Y ** 2) / (sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)  # (out, in, kernel_size)

    # 🔑 关键：将 sigma 为 0 对应的核全部置为 0
    if zero_mask.any():
        Envelop = Envelop * (~zero_mask).float()

    # Construct corresponding Gabor kernels
    Freq = Freq_Y * coords_Y  # (out, in, kernel_size)

    if g_type == 'Re':
        kernels = Envelop * torch.cos(Freq + phase)
    elif g_type == 'Im':
        kernels = Envelop * torch.sin(Freq + phase)
    else:
        raise ValueError('Error Gabor filter type')

    return kernels.unsqueeze(2).permute(0, 1, 3, 2)


def get_1DGabor_kernels_y_ab_scattering_phase(theta: torch.Tensor, omega: torch.Tensor, sigma: torch.Tensor,
                          kernel_size: int, in_channels=103, out_channels=16, device='cuda:0', g_type='Im'):
    sigma = sigma.expand(-1, in_channels)  #ablation: no channel sharing
    # sigma = sigma.T                #ablation: no scattering
    # phase = phase.expand(out_channels, -1)
    omega = omega.expand(-1, in_channels)
    # omega = omega.T                #ablation: no scattering
    theta = theta.expand(-1, in_channels)
    theta = theta.T                #ablation: no scattering
    sigma = sigma.unsqueeze(-1)  # [out, in, 1]
    # phase = phase.unsqueeze(-1)  # [out, in, 1]

    # Freq_X = omega * torch.cos(theta)  # [out, in]
    Freq_Y = omega * torch.sin(theta)  # [out, in]
    Freq_Y = Freq_Y.unsqueeze(-1)  # [out, in, 1]

    # Create the Euclidean coordinates of the kernel
    coords = L2_grid(kernel_size)[np.newaxis, np.newaxis, :]  # (1, 1, kernel_size)
    coords_Y = torch.tensor(coords, dtype=torch.float32, device=device)

    # Envelop construction
    Envelop = torch.exp(-0.5 * (coords_Y ** 2) / (sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)  # (out, in, kernel_size)

    # Construct corresponding Gabor kernels
    Freq = Freq_Y * coords_Y  # (out, in, kernel_size)

    if g_type == 'Re':
        kernels = Envelop * torch.cos(Freq)
    elif g_type == 'Im':
        kernels = Envelop * torch.sin(Freq)
    else:
        raise ValueError('Error Gabor filter type')

    return kernels.unsqueeze(2).permute(0, 1, 3, 2)

def get_1DGabor_kernels_y_ab_scattering_others(theta: torch.Tensor, omega: torch.Tensor, sigma: torch.Tensor, phase: torch.Tensor,
                          kernel_size: int, in_channels=103, out_channels=16, device='cuda:0', g_type='Im'):
    sigma = sigma.expand(-1, in_channels)  #ablation: no channel sharing
    # sigma = sigma.T                #ablation: no scattering
    # phase = phase.expand(out_channels, -1)
    omega = omega.expand(-1, in_channels)
    omega = omega.T                #ablation: no scattering
    theta = theta.expand(-1, in_channels)
    theta = theta.T                #ablation: no scattering
    sigma = sigma.unsqueeze(-1)  # [out, in, 1]
    phase = phase.unsqueeze(-1)  # [out, in, 1]

    # Freq_X = omega * torch.cos(theta)  # [out, in]
    Freq_Y = omega * torch.sin(theta)  # [out, in]
    Freq_Y = Freq_Y.unsqueeze(-1)  # [out, in, 1]

    # Create the Euclidean coordinates of the kernel
    coords = L2_grid(kernel_size)[np.newaxis, np.newaxis, :]  # (1, 1, kernel_size)
    coords_Y = torch.tensor(coords, dtype=torch.float32, device=device)

    # Envelop construction
    Envelop = torch.exp(-0.5 * (coords_Y ** 2) / (sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)  # (out, in, kernel_size)

    # Construct corresponding Gabor kernels
    Freq = Freq_Y * coords_Y  # (out, in, kernel_size)

    if g_type == 'Re':
        kernels = Envelop * torch.cos(Freq + phase)
    elif g_type == 'Im':
        kernels = Envelop * torch.sin(Freq + phase)
    else:
        raise ValueError('Error Gabor filter type')

    return kernels.unsqueeze(2).permute(0, 1, 3, 2)

def get_DeGCMkernels_y_ab_scattering_others(
    theta: torch.Tensor,
    omega: torch.Tensor,
    sigma: torch.Tensor,
    phase: torch.Tensor,
    kernel_size: int,
    in_channels: int = 103,
    out_channels: int = 16,
    device: Union[str, torch.device] = 'cuda:0',
    g_type: bool = True
) -> torch.Tensor:
    sigma = sigma.expand(-1, in_channels)
    sigma = sigma.T                #ablation: no scattering
    # phase = phase.expand(out_channels, -1)
    omega = omega.expand(-1, in_channels)
    omega = omega.T                #ablation: no scattering
    theta = theta.expand(-1, in_channels)
    theta = theta.T                #ablation: no scattering
    sigma = sigma.unsqueeze(-1)  # [out, in, 1]
    phase = phase.unsqueeze(-1)  # [out, in, 1]

    # Freq_X = omega * torch.cos(theta)  # [out, in]
    Freq_Y = omega * torch.sin(theta)  # [out, in]
    Freq_Y = Freq_Y.unsqueeze(-1)  # [out, in, 1]

    # Create the Euclidean coordinates of the kernel
    # coords = L2_grid(kernel_size)[np.newaxis, np.newaxis, :]  # (1, 1, kernel_size)
    # coords_Y = torch.tensor(coords, dtype=torch.float32, device=device)
    coords = L2_grid1(kernel_size).unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size)
    coords_Y = coords.to(device='cuda', dtype=torch.float32)

    # Envelop construction
    # Envelop = torch.exp(-0.5 * (coords_Y ** 2) / (sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)  # (out, in, kernel_size)
    Envelop = torch.exp(-0.5 * (coords_Y ** 2) / (sigma ** 2)) / (torch.sqrt(2 * math.pi) * sigma)

    # Construct corresponding Gabor kernels
    Freq = Freq_Y * coords_Y  # (out, in, kernel_size)

    if g_type:
        kernels = Envelop * torch.cos(Freq + phase)
    else:
        kernels = Envelop * torch.sin(Freq + phase)

    return kernels.unsqueeze(2).permute(0, 1, 3, 2)


def get_DeGCMkernels_x(
    theta: torch.Tensor,
    omega: torch.Tensor,
    sigma: torch.Tensor,
    phase: torch.Tensor,
    kernel_size: int,
    in_channels: int = 103,
    out_channels: int = 16,
    device: Union[str, torch.device] = 'cuda:0',
    g_type: bool = True
) -> torch.Tensor:
    # sh = list(theta.shape)              # [out,in]
    # 在索引为-1上增加一个维度，二维 -> 三维
    sigma = sigma.expand(-1, in_channels)
    # phase = phase.expand(out_channels, -1)
    omega = omega.expand(-1, in_channels)
    theta = theta.expand(-1, in_channels)

    sigma = torch.unsqueeze(sigma, -1)  # [out,in, 1]
    phase = torch.unsqueeze(phase, -1)  # [out,in, 1]

    Freq_X = (omega * torch.cos(theta)).unsqueeze(-1).to('cuda')  # [out, in, 1]

    # 创建欧几里得坐标
    # coords = L2_grid(kernel_size)[np.newaxis, np.newaxis, :]  # (1, 1, kernel_size)
    # coords_X = torch.tensor(coords, dtype=torch.float32, device=device)
    coords = L2_grid1(kernel_size).unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size)
    coords_X = coords.to(device='cuda', dtype=torch.float32)

    # 计算包络
    # Envelop = torch.exp(-0.5 * (coords_X ** 2) / (sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)  # (out, in, kernel_size)
    Envelop = torch.exp(-0.5 * (coords_X ** 2) / (sigma ** 2)) / (torch.sqrt(2 * math.pi) * sigma)

    # 构建Gabor核
    Freq = Freq_X * coords_X  # (out, in, kernel_size)
    if g_type:
        kernels = Envelop * torch.cos(Freq + phase)
    else:
        kernels = Envelop * torch.sin(Freq + phase)

    return kernels.unsqueeze(2)  # 返回结果


def get_DeGCMkernels_y(
    theta: torch.Tensor,
    omega: torch.Tensor,
    sigma: torch.Tensor,
    phase: torch.Tensor,
    kernel_size: int,
    in_channels: int = 103,
    out_channels: int = 16,
    device: Union[str, torch.device] = 'cuda:0',
    g_type: bool = True
) -> torch.Tensor:
    sigma = sigma.expand(-1, in_channels)
    # phase = phase.expand(out_channels, -1)
    omega = omega.expand(-1, in_channels)
    theta = theta.expand(-1, in_channels)
    sigma = sigma.unsqueeze(-1)  # [out, in, 1]
    phase = phase.unsqueeze(-1)  # [out, in, 1]

    # Freq_X = omega * torch.cos(theta)  # [out, in]
    Freq_Y = omega * torch.sin(theta)  # [out, in]
    Freq_Y = Freq_Y.unsqueeze(-1)  # [out, in, 1]

    # Create the Euclidean coordinates of the kernel
    # coords = L2_grid(kernel_size)[np.newaxis, np.newaxis, :]  # (1, 1, kernel_size)
    # coords_Y = torch.tensor(coords, dtype=torch.float32, device=device)
    coords = L2_grid1(kernel_size).unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size)
    coords_Y = coords.to(device='cuda', dtype=torch.float32)

    # Envelop construction
    # Envelop = torch.exp(-0.5 * (coords_Y ** 2) / (sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)  # (out, in, kernel_size)
    Envelop = torch.exp(-0.5 * (coords_Y ** 2) / (sigma ** 2)) / (torch.sqrt(2 * math.pi) * sigma)

    # Construct corresponding Gabor kernels
    Freq = Freq_Y * coords_Y  # (out, in, kernel_size)

    if g_type:
        kernels = Envelop * torch.cos(Freq + phase)
    else:
        kernels = Envelop * torch.sin(Freq + phase)

    return kernels.unsqueeze(2).permute(0, 1, 3, 2)

def L2_grid(kernel_size):
    """
    get the Euclidean coordinates of the kernel 核的欧氏坐标

    :param kernel_size: size of the target kernel

    :return: coordinates of the target kernel
    """
    # Get neighbourhoods
    center = kernel_size // 2
    lin = np.arange(kernel_size)
    # numpy.meshgrid()函数可以让我们快速生成坐标矩阵X,Y  #J,I是坐标矩阵  #输入的x,y,就是网格点的横纵坐标列向量（非矩阵）
    I = np.arange(len(lin))  # J-X I-Y
    I -= center
    return I


# 参数的初始化
# 参数的形状都为(n_out, n_in)=(n_theta x n_omega,band)
def get_theta(n_in, n_theta, n_omega, even_initial=False, device='cuda:0', requires_grad=False):
    """
    Get the list of theta

    :param n_in: number of the channels of inputs (int)
    :param n_theta: number of the initialization values of theta (int)
    :param n_omega: number of the initialization values of theta (int)
    :param even_initial: whether adopt the even initialization strategy (boolean) 是否采用均匀初始化策略(布尔值)
    :param device
    :param requires_grad: whether to use gradients (boolean)
    :return: a list of theta
    """
    if even_initial:
        # np.linspace主要用来创建等差数列 # endpoint：True则包含stop；False则不包含stop # 转化为一列
        init = np.linspace(0, 1, num=n_theta, endpoint=False, dtype=np.float32).reshape((-1, 1)) * np.pi
        # np.tile(a,(2,1))第一个参数为Y轴扩大倍数，第二个为X轴扩大倍数
        init = np.tile(init, (n_omega, 1))
        init = np.tile(init, (1, n_in))
    else:
        # n_out = n_theta * n_omega
        # init = np.random.rand(n_out, n_in) * np.pi#随机分布

        n_out = n_theta * n_omega
        init = np.random.randn(n_out, n_in)*np.pi  # 均值 0，标准差 pi

    # theta最后的形状为(n_out, n_in)
    return torch.tensor(init, dtype=torch.float32, device=device, requires_grad=requires_grad)


def get_omega(n_in, n_theta, n_omega, even_initial=False, mean=0, device='cuda:0', requires_grad=False):
    """
    Get the list of omega

    :param n_in: number of the channels of inputs (int)
    :param n_theta: number of the initialization values of theta (int)
    :param n_omega: number of the initialization values of theta (int)
    :param even_initial: whether adopt the even initialization strategy (boolean)
    :param mean: the mean of normal distribution in the random initialization strategy
    :param name: (default: omega)
    :param device
    :param requires_grad

    :return: a list of omega
    """
    if even_initial:
        # np.logspace() 对数等比数列
        init = np.logspace(0, n_omega - 1, num=n_omega, base=1 / 2) * (np.pi / 2)
        init = init[:, np.newaxis]  # [4,1]
        init = np.tile(init, [1, n_theta])  # [4,4]
        init = np.reshape(init, [1, -1])  # [1,16]
        init = np.tile(init, (n_in, 1))  # [n_in,16]=[n_in,n_out]
        # 最后形状为(n_out, n_in)
        return torch.tensor(init.transpose((1, 0)), dtype=torch.float32, device=device, requires_grad=requires_grad)
    else:
        # n_out = n_theta * n_omega
        # init = np.random.rand(n_out, n_in) * (2 * np.pi)
        # #omega最后的形状为(n_out, n_in)
        # return torch.tensor(init, dtype=torch.float32, device=device, requires_grad=requires_grad)
        # n_out = n_omega * n_theta
        # stddev = np.pi / 8
        # # 该函数返回从单独的正态分布中提取的随机数的张量，该正态分布的均值是mean，标准差是std
        # return torch.normal(mean=mean, std=stddev, size=[n_out, n_in], dtype=torch.float32, device=device,
        #                     requires_grad=requires_grad)
        n_out = n_theta * n_omega
        # 随机均匀分布在 [0, π]
        # init = np.random.uniform(low=0.0, high=np.pi, size=(n_out, n_in))
        init = np.random.randn(n_out, n_in) * np.pi  # 均值 0，标准差 pi

        return torch.tensor(init, dtype=torch.float32, device=device, requires_grad=requires_grad)


def get_sigma(n_in, n_out, kernel_size=5, even_initial=False, mean=0, device='cuda:0', requires_grad=False):
    """
    Get the list of sigma

    :param n_in: number of the channels of inputs (int)
    :param n_out: number of the channels of outputs (int)
    :param kernel_size: size of kernels
    :param even_initial: whether adopt the even initialization strategy
    :param mean: the mean of normal distribution in the random initialization strategy
    :param device
    :param requires_grad
    :return: a list of sigma
    """
    if even_initial:
        return torch.ones(n_out, n_in, dtype=torch.float32, device=device,
                          requires_grad=requires_grad) * kernel_size / 8
    else:
        # 随机均匀分布在 [0, π]
        # init = np.random.uniform(low=0.0, high=np.pi, size=(n_out, n_in))
        init = np.random.randn(n_out, n_in) * np.pi  # 均值 0，标准差 pi
        return torch.tensor(init, dtype=torch.float32, device=device, requires_grad=requires_grad)
        # init = np.random.rand(n_out, n_in) * (2 * np.pi)
        # # omega最后的形状为(n_out, n_in)
        # return torch.tensor(init, dtype=torch.float32, device=device, requires_grad=requires_grad)
        # stddev = (5 / 4) * (1 / 2)
        # return torch.normal(mean=mean, std=stddev, size=[n_out, n_in], dtype=torch.float32, device=device,
        #                     requires_grad=requires_grad)


def get_phase(n_in, n_out, even_initial=False, device='cuda:0', requires_grad=False):
    """
    P
    Get the list of phase offsets

    :param n_in: number of the channels of inputs (int)
    :param n_out: number of the channels of outputs (int)
    :param even_initial: whether adopt the even initialization strategy 是否采用均匀初始化策略
    :param name: (default: P)
    :param device
    :param requires_grad
    :return: a list of phase offsets
    """
    if even_initial:
        # initialization corresponding to each output
        init = np.random.uniform(low=0.0, high=np.pi, size=(n_out, n_in))
        return torch.tensor(init, dtype=torch.float32, device=device, requires_grad=requires_grad) # 保持和 torch.float32 一致
    else:
        # torch.randn标准正态分布
        return torch.randn(n_out, n_in, dtype=torch.float32, device=device, requires_grad=requires_grad) * np.pi


def L2_grid1(kernel_size:int):
    """
    获取核的欧氏坐标

    :param kernel_size: 目标核的大小
    :return: 目标核的坐标
    """
    # 获取邻域
    center = kernel_size // 2
    lin = torch.arange(kernel_size, dtype=torch.float32)  # 使用 torch.arange 替代 np.arange
    I = torch.arange(len(lin), dtype=torch.float32)  # 使用 torch.arange 生成 Y 方向的坐标
    I -= center  # 将中心坐标移到 0

    return I


if __name__ == '__main__':
    a = get_2DGabor_kernels(
        theta=get_theta(3, 4, 4, device='cuda', even_initial=True),
        omega=get_omega(3, 4, 4, device='cuda', even_initial=True),
        sigma=get_sigma(3, 16, device='cuda', even_initial=True),
        phase=get_phase(3, 16, device='cuda', even_initial=True),
        kernel_size=5,
        device='cuda'
    )
    # print(a)
    print(a.shape, a.requires_grad)
    # torch.Size([16, 3, 5, 5]) False
    # reture=[out,in,kernel_size,kernel_size]
    # requires_grad=False时表示不需要计算梯度
