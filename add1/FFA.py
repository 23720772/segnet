import torch.nn as nn
import torch
# https://arxiv.org/pdf/1911.07559

'''
FFA-Net：用于单图像去雾的特征融合注意力网络  (AAAI 人工智能顶会）
即插即用模块：FFA特征融合注意力 
在本文中，我们提出了一种端到端的特征融合注意力FFA模块用来图像去雾。
FFA-Net架构由三个关键组件组成：
    1）考虑到不同的通道特征包含完全不同的加权信息，以及不同图像像素上的雾度分布不均匀，
    一种新的特征注意力（FA）模块将通道注意力与像素注意力机制相结合。
    FA对不同的特征和像素进行不平等处理，为处理不同类型的信息提供了额外的灵活性，扩展了CNNs的表征能力。
    
    2）基本块结构由局部残差学习和特征注意力组成，局部残差学习允许舍弃不重要的信息，
    如薄雾区或低频信息通过多个局部残差连接被绕过， 让主网络架构专注于更有效的信息。
    
    3）基于注意力的不同层次特征融合（FFA）结构，特征权重自适应地从特征注意力（FA）模块中学习，
    赋予重要特征更多的权重。这种结构还可以保留浅层的信息，并将其传递到深层。
    
适用于：图像增强，暗光增强，图像去雾，图像去噪，目标检测，图像分割，图像分类等所有CV2维任务通用的即插即用注意力模块
'''
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
class PALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y
class CALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y
class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super().__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super().__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res
class FFA(nn.Module):
    def __init__(self,inchannel= 3, blocks=1, conv=default_conv):
        super().__init__()
        self.gps = inchannel
        self.dim = 8
        kernel_size = 3
        pre_process = [conv(self.gps, self.dim, kernel_size)]
        # assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim *3, self.dim // 4, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 4, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.palayer = PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, self.gps, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        o = torch.cat([res1, res2, res3], dim=1)
        w = self.ca(o)
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        out = self.palayer(out)
        x = self.post(out)
        return x + x1
# 输入 N C H W,  输出 N C H W
if __name__ == "__main__":
    input =  torch.randn(1, 32, 640, 640)
    model = FFA(inchannel=32)
    output = model(input)
    print('input_size:',input.size())
    print('output_size:',output.size())