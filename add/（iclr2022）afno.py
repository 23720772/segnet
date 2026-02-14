# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """

    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        # 检查hidden_size是否可以被num_blocks整除
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        # 初始化参数
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x, spatial_size=None):
        bias = x

        # 获取输入数据的类型
        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        # 获取输入数据的尺寸
        if spatial_size == None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size

        # 将输入数据reshape为(B, H, W, C)
        x = x.reshape(B, H, W, C)
        # 对输入数据进行二维快速傅里叶变换
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        # 将输入数据reshape为(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        # 初始化输出数据
        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # 计算总频数和保留的频数
        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # 计算o1_real和o1_imag
        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        # 计算o2_real和o2_imag
        o2_real[:, :, :kept_modes] = (
                torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
                torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
                self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
                torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
                torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
                self.b2[1]
        )

        # 将o2_real和o2_imag堆叠起来
        x = torch.stack([o2_real, o2_imag], dim=-1)
        # 对x进行软阈值化
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        # 将x转换为复数
        x = torch.view_as_complex(x)
        # 将x reshape为(B, x.shape[1], x.shape[2], C)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        # 对x进行二维逆快速傅里叶变换
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        # 将x reshape为(B, N, C)
        x = x.reshape(B, N, C)
        # 将x转换为原始数据类型
        x = x.type(dtype)
        # 返回x加上偏置
        return x + bias


#  输入 B N C ,  输出 B N C
if __name__ == '__main__':
    input = torch.randn(3, 64 * 64, 32).cuda()
    model = AFNO2D(32,num_blocks=8).cuda()  # 通道数为num_blocks=8的倍数
    output = model(input)
    print(output.shape)
