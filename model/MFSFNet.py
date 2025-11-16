import copy
import datetime
import os
import pandas as pd
import torchvision
import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, ResNet34_Weights



__all__ = ["ResNet50", "ResNet34"]

from model.isdhead import RAF


def ResNet50():
    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    return resnet50


def ResNet34():
    resnet34 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    return resnet34

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )
class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(256, 256), mode='bilinear', align_corners=False)
        return feat


class RCA(nn.Module):
    def __init__(self, inp, kernel_size=1, ratio=1, band_kernel_size=11, dw_size=(1, 1), padding=(0, 0), stride=1,
                 square_kernel_size=2, relu=True):
        super(RCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc = inp // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )

    def sge(self, x):
        # [N, D, C, 1]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w  # .repeat(1,1,1,x_w.shape[-1])
        ge = self.excite(x_gather)  # [N, 1, C, 1]

        return ge

    def forward(self, x):
        loc = self.dwconv_hw(x)
        att = self.sge(x)
        out = att * loc

        return out


class MFSF4(nn.Module):
    def __init__(self, in_dim=128):
        super(MFSF4, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * in_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim),
            nn.PReLU()
        )
        self.token_mixer = RCA(in_dim, band_kernel_size=11, square_kernel_size=3, ratio=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.learnable_filter = nn.Parameter(torch.randn(in_dim, 1, 1))

    def forward(self, input_rgb, input_dsm):
        add_fusion = input_rgb + input_dsm

        # 频域变换
        x_freq = torch.fft.fft2(add_fusion)
        x_freq = torch.fft.fftshift(x_freq)

        # Hadamard乘法和动态滤波
        x_freq = x_freq * self.learnable_filter

        # 逆频域变换
        x_freq = torch.fft.ifftshift(x_freq)
        x_out = torch.fft.ifft2(x_freq).real

        d_1 = torch.cat((input_rgb, x_out, input_dsm), dim=1)
        fusion = self.fuse(d_1)
        fusion = self.token_mixer(fusion)

        return fusion + input_rgb, input_dsm


class PFAM3(nn.Module):
    def __init__(self, dim, in_dim):
        super(PFAM3, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim), nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.learnable_filter2 = nn.Parameter(torch.randn(down_dim, 1, 1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.learnable_filter3 = nn.Parameter(torch.randn(down_dim, 1, 1))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))
        self.learnable_filter4 = nn.Parameter(torch.randn(down_dim, 1, 1))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.learnable_filter5 = nn.Parameter(torch.randn(down_dim, 1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=in_dim // 16, out_channels=in_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        conv1 = self.conv1(x)

        conv2 = self.conv2(x)
        # m_batchsize, C, height, width = conv2.size()
        # proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)
        # energy2 = torch.bmm(proj_query2, proj_key2)
        # attention2 = self.softmax(energy2)
        # proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        # out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        # out2 = out2.view(m_batchsize, C, height, width)
        # 频域变换
        x_freq = torch.fft.fft2(conv2)
        x_freq = torch.fft.fftshift(x_freq)

        # Hadamard乘法和动态滤波
        x_freq = x_freq * self.learnable_filter2

        # 逆频域变换
        x_freq = torch.fft.ifftshift(x_freq)
        x_out = torch.fft.ifft2(x_freq).real
        # out2 = self.gamma2 * out2 + conv2 + x_out
        out2 = x_out + conv2
        conv3 = self.conv3(x)
        # m_batchsize, C, height, width = conv3.size()
        # proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        # energy3 = torch.bmm(proj_query3, proj_key3)
        # attention3 = self.softmax(energy3)
        # proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        # out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        # out3 = out3.view(m_batchsize, C, height, width)
        # 频域变换
        x_freq = torch.fft.fft2(conv3)
        x_freq = torch.fft.fftshift(x_freq)

        # Hadamard乘法和动态滤波
        x_freq = x_freq * self.learnable_filter3

        # 逆频域变换
        x_freq = torch.fft.ifftshift(x_freq)
        x_out = torch.fft.ifft2(x_freq).real
        # out3 = self.gamma3 * out3 + conv3 + x_out
        out3 = x_out + conv3
        conv4 = self.conv4(x)
        # m_batchsize, C, height, width = conv4.size()
        # proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        # energy4 = torch.bmm(proj_query4, proj_key4)
        # attention4 = self.softmax(energy4)
        # proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        # out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        # out4 = out4.view(m_batchsize, C, height, width)
        # 频域变换
        x_freq = torch.fft.fft2(conv4)
        x_freq = torch.fft.fftshift(x_freq)

        # Hadamard乘法和动态滤波
        x_freq = x_freq * self.learnable_filter4

        # 逆频域变换
        x_freq = torch.fft.ifftshift(x_freq)
        x_out = torch.fft.ifft2(x_freq).real
        # out4 = self.gamma4 * out4 + conv4 + x_out
        out4 = x_out + conv4

        x_freq = torch.fft.fft2(conv1)
        x_freq = torch.fft.fftshift(x_freq)

        # Hadamard乘法和动态滤波
        x_freq = x_freq * self.learnable_filter5

        # 逆频域变换
        x_freq = torch.fft.ifftshift(x_freq)
        conv5 = torch.fft.ifft2(x_freq).real

        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * x

        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1)) + c_attn



class MFSFNet(nn.Module):
    def __init__(self, num_classes=8, RGB_flag=False):
        super(MFSFNet, self).__init__()
        resnet50 = ResNet50()
        self.RGB_flag = RGB_flag
        self.dsm_conv0 = nn.Conv2d(1, 3, kernel_size=1)
        if self.RGB_flag:
            self.rgb_conv1 = resnet50.conv1
        else:
            # 获取原 conv1 的权重和偏置
            original_conv1_weight = resnet50.conv1.weight.data  # 形状 [64, 3, 7, 7]

            # 创建新的卷积层 conv2（输入 4 通道，输出 64 通道）
            rgb_conv1 = nn.Conv2d(
                in_channels=4,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False if resnet50.conv1.bias is None else True
            )

            # 初始化新卷积层的权重
            with torch.no_grad():
                # 新权重形状 [64, 4, 7, 7]
                new_weight = torch.zeros_like(rgb_conv1.weight.data)

                # 将原 conv1 的权重复制到前 3 个通道
                new_weight[:, :3, :, :] = original_conv1_weight

                # 方法 2：均值填充（可选）
                new_weight[:, 3, :, :] = original_conv1_weight.mean(dim=1)  # 取前 3 个通道的均值

                # 将新权重赋值给 conv2
                rgb_conv1.weight.data = new_weight

                self.rgb_conv1 = rgb_conv1

        self.dsm_conv1 = copy.deepcopy(resnet50.conv1)

        self.rgb_bn1 = resnet50.bn1
        self.dsm_bn1 = copy.deepcopy(resnet50.bn1)

        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool

        self.rgb_layer1 = resnet50.layer1
        self.dsm_layer1 = copy.deepcopy(resnet50.layer1)

        self.rgb_layer2 = resnet50.layer2
        self.dsm_layer2 = copy.deepcopy(resnet50.layer2)

        self.rgb_layer3 = resnet50.layer3
        self.dsm_layer3 = copy.deepcopy(resnet50.layer3)

        self.se_layer0 = MFSF4(64)
        self.se_layer1 = MFSF4(256)
        self.se_layer2 = MFSF4(512)
        self.se_layer3 = MFSF4(1024)

        # 解码器
        self.conv_more = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        self.MSTransformer1 = PFAM3(in_dim=512, dim=512)
        self.MSTransformer2 = PFAM3(in_dim=256, dim=256)
        self.MSTransformer3 = PFAM3(in_dim=128, dim=128)

        ##############layer1##################
        self.layer_1_1 = Conv2dReLU(1024, 256, kernel_size=3, padding=1)
        self.layer_1_2 = Conv2dReLU(256, 256, kernel_size=3, padding=1)

        ##############layer2##################
        self.layer_2_1 = Conv2dReLU(512, 128, kernel_size=3, padding=1)
        self.layer_2_2 = Conv2dReLU(128, 128, kernel_size=3, padding=1)

        ##############layer3##################
        self.layer_3_1 = Conv2dReLU(192, 64, kernel_size=3, padding=1)
        self.layer_3_2 = Conv2dReLU(64, 64, kernel_size=3, padding=1)

        ##############layer4##################
        self.layer_4_1 = Conv2dReLU(64, 16, kernel_size=3, padding=1)
        self.layer_4_2 = Conv2dReLU(16, 16, kernel_size=3, padding=1)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # 分割头
        self.segmentationHead = SegmentationHead(16, num_classes)



    def forward(self, input_rgb, input_dsm):
        SE = True
        features = []
        input_dsm = self.dsm_conv0(input_dsm)  # dsm通道变3

        input_rgb0 = self.relu(self.rgb_bn1(self.rgb_conv1(input_rgb)))
        input_dsm0 = self.relu(self.dsm_bn1(self.dsm_conv1(input_dsm)))
        if SE:
            fusion0, input_dsm0 = self.se_layer0(input_rgb0, input_dsm0)
        else:
            fusion0 = input_rgb0 + input_dsm0
        features.append(fusion0)
        input_rgb = self.maxpool(fusion0)
        input_dsm = self.maxpool(input_dsm0)

        input_rgb1 = self.rgb_layer1(input_rgb)
        input_dsm1 = self.dsm_layer1(input_dsm)
        if SE:
            fusion1, input_dsm1 = self.se_layer1(input_rgb1, input_dsm1)
        else:
            fusion1 = input_rgb1 + input_dsm1
        features.append(fusion1)

        input_rgb2 = self.rgb_layer2(fusion1)
        input_dsm2 = self.dsm_layer2(input_dsm1)
        if SE:
            fusion2, input_dsm2 = self.se_layer2(input_rgb2, input_dsm2)
        else:
            fusion2 = input_rgb2 + input_dsm2
        features.append(fusion2)

        # 深层 RGB
        input_rgb3 = self.rgb_layer3(fusion2)
        input_dsm3 = self.dsm_layer3(input_dsm2)
        if SE:
            fusion3, input_dsm3 = self.se_layer3(input_rgb3, input_dsm3)
        else:
            fusion3 = input_rgb3 + input_dsm3
        ############DECODER#########
        x = self.conv_more(fusion3)
        x = self.MSTransformer1(x)
        x = self.up(x)
        #############layer1###########
        # x = self.raf(x, fusion2)
        x = torch.cat([x, fusion2], dim=1)
        x = self.layer_1_1(x)
        x = self.layer_1_2(x)
        #############layer2###########
        # x = self.MSTransformer2(x)
        x = self.up(x)
        x = torch.cat([x, fusion1], dim=1)  # 256 64 64
        x = self.layer_2_1(x)
        x = self.layer_2_2(x)
        #############layer3###########
        # x = self.MSTransformer3(x)
        x = self.up(x)
        x = torch.cat([x, fusion0], dim=1)
        x = self.layer_3_1(x)
        x = self.layer_3_2(x)
        #############layer4###########
        x = self.up(x)
        x = self.layer_4_1(x)
        x = self.layer_4_2(x)

        ###############SegmentationHead################
        logits = self.segmentationHead(x)
        return logits


if __name__ == "__main__":
    from thop import profile, clever_format

    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 1, 256, 256)
    net = MFSFNet(RGB_flag=True)

    # net.load_state_dict(torch.load('../checkpoint/WHU_RGBN_SAR_Resnet50_SKRCM2_MHSA_1_FreMX_epoch_50_79.96870759887616'),
    #                     strict=False)
    flops, params = profile(net, inputs=(x, y))
    macs, params = clever_format([flops, params], "%.3f")
    print("FLOPs:", macs)
    print("params:", params)
    # result_dict = {}
    # result_dict["时间"] = datetime.datetime.now()
    # result_dict["模型格式"] = "baseline_Resnet50_CSAM_CMSGM"
    # result_dict["FLOPs"] = macs
    # result_dict["params"] = params
    # # 创建一个DataFrame
    # df = pd.DataFrame([result_dict])
    # # 将DataFrame写入CSV文件
    # # 获取当前脚本（utils.py）所在的目录
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # # 构造相对于当前脚本的results文件夹路径
    # csv_filename = os.path.join(current_dir, 'FLOPs.csv')
    # # 检查CSV文件是否存在
    # if not os.path.exists(csv_filename):
    #     # 如果文件不存在，保存第一次数据并包含列名
    #     df.to_csv(csv_filename, index=False)
    # else:
    #     # 如果文件存在，以追加模式打开文件并写入后续数据（不包含列名）
    #     with open(csv_filename, 'a', newline='') as f:
    #         df.to_csv(f, header=False, index=False)


# import time
# from torch.autograd import Variable
# # Example usage
# if __name__ == "__main__":
#     # Define a sample model and input tensor
#     iterations = 100
#     x = Variable(torch.randn(1, 3, 256, 256).cuda())
#     y = Variable(torch.randn(1, 1, 256, 256).cuda())
#     model = CMSFNet(RGB_flag=True).cuda()
#
#     model.eval()
#     with torch.no_grad():
#         # Warm-up to stabilize performance
#         for _ in range(10):
#             _ = model(x, y)
#
#         start_time = time.time()
#
#         # Run the model for the specified number of iterations
#         for _ in range(iterations):
#             _ = model(x, y)
#
#         end_time = time.time()
#
#     # Calculate FPS
#     total_time = end_time - start_time
#     fps = iterations / total_time
#     print(f"FPS: {fps:.2f}")
