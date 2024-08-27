import os
import numpy as np
import torch
import torch.nn as nn
from layer import *
from util import *


# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#
#         ## First Blue arrow
#         self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
#         self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
#
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#
#         self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
#         self.enc2_2 = CBR2d(in_channels=128, out_channels=128)
#
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#
#         self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
#         self.enc3_2 = CBR2d(in_channels=256, out_channels=256)
#
#         self.pool3 = nn.MaxPool2d(kernel_size=2)
#
#         self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
#         self.enc4_2 = CBR2d(in_channels=512, out_channels=512)
#
#         self.pool4 = nn.MaxPool2d(kernel_size=2)
#
#         self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)
#
#         # Expansive path
#         self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)
#
#         self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
#                                           kernel_size=2, stride=2, padding=0, bias=True)
#
#         self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
#         self.dec4_1 = CBR2d(in_channels=512, out_channels=256)
#
#         self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
#                                           kernel_size=2, stride=2, padding=0, bias=True)
#
#         self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
#         self.dec3_1 = CBR2d(in_channels=256, out_channels=128)
#
#         self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
#                                           kernel_size=2, stride=2, padding=0, bias=True)
#
#         self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
#         self.dec2_1 = CBR2d(in_channels=128, out_channels=64)
#
#         self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
#                                           kernel_size=2, stride=2, padding=0, bias=True)
#
#         self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
#         self.dec1_1 = CBR2d(in_channels=64, out_channels=64)
#
#         self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
#
#     def forward(self, x):
#         enc1_1 = self.enc1_1(x)
#         enc1_2 = self.enc1_2(enc1_1)
#         pool1 = self.pool1(enc1_2)
#
#         enc2_1 = self.enc2_1(pool1)
#         enc2_2 = self.enc2_2(enc2_1)
#         pool2 = self.pool2(enc2_2)
#
#         enc3_1 = self.enc3_1(pool2)
#         enc3_2 = self.enc3_2(enc3_1)
#         pool3 = self.pool3(enc3_2)
#
#         enc4_1 = self.enc4_1(pool3)
#         enc4_2 = self.enc4_2(enc4_1)
#         pool4 = self.pool4(enc4_2)
#
#         enc5_1 = self.enc5_1(pool4)
#
#         dec5_1 = self.dec5_1(enc5_1)
#
#         unpool4 = self.unpool4(dec5_1)
#         cat4 = torch.cat((unpool4, enc4_2), dim=1)
#         dec4_2 = self.dec4_2(cat4)
#         dec4_1 = self.dec4_1(dec4_2)
#
#         unpool3 = self.unpool3(dec4_1)
#         cat3 = torch.cat((unpool3, enc3_2), dim=1)
#         dec3_2 = self.dec3_2(cat3)
#         dec3_1 = self.dec3_1(dec3_2)
#
#         unpool2 = self.unpool2(dec3_1)
#         cat2 = torch.cat((unpool2, enc2_2), dim=1)
#         dec2_2 = self.dec2_2(cat2)
#         dec2_1 = self.dec2_1(dec2_2)
#
#         unpool1 = self.unpool1(dec2_1)
#         cat1 = torch.cat((unpool1, enc1_2), dim=1)
#         dec1_2 = self.dec1_2(cat1)
#         dec1_1 = self.dec1_1(dec1_2)
#
#         x = self.fc(dec1_1)
#
#         return x

# class ResUnet(nn.Module):
#     def __init__(self, channel=1, filters=[64, 128, 256, 512]):
#         super(ResUnet, self).__init__()
#
#         self.input_layer = nn.Sequential(
#             nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
#             nn.BatchNorm2d(filters[0]),
#             nn.ReLU(),
#             nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
#         )
#         self.input_skip = nn.Sequential(
#             nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
#         )
#
#         self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
#         self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
#
#         self.bridge = ResidualConv(filters[2], filters[3], 2, 1)
#
#         self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
#         self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)
#
#         self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
#         self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)
#
#         self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
#         self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)
#
#         self.output_layer = nn.Sequential(
#             nn.Conv2d(filters[0], 1, 1, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         # Encode
#         x1 = self.input_layer(x) + self.input_skip(x)
#         x2 = self.residual_conv_1(x1)
#         x3 = self.residual_conv_2(x2)
#         # Bridge
#         x4 = self.bridge(x3)
#         # Decode
#         x4 = self.upsample_1(x4)
#         x5 = torch.cat([x4, x3], dim=1)
#
#         x6 = self.up_residual_conv1(x5)
#
#         x6 = self.upsample_2(x6)
#         x7 = torch.cat([x6, x2], dim=1)
#
#         x8 = self.up_residual_conv2(x7)
#
#         x8 = self.upsample_3(x8)
#         x9 = torch.cat([x8, x1], dim=1)
#
#         x10 = self.up_residual_conv3(x9)
#
#         output = self.output_layer(x10)
#
#         return output

# class R2U_Net(nn.Module):
#     def __init__(self, img_ch=1, output_ch=1, t=2):
#         super(R2U_Net, self).__init__()
#
#         n1 = 64
#         filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
#
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.Upsample = nn.Upsample(scale_factor=2)
#
#         self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)
#
#         self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)
#
#         self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)
#
#         self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)
#
#         self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)
#
#         self.Up5 = up_conv(filters[4], filters[3])
#         self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)
#
#         self.Up4 = up_conv(filters[3], filters[2])
#         self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)
#
#         self.Up3 = up_conv(filters[2], filters[1])
#         self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)
#
#         self.Up2 = up_conv(filters[1], filters[0])
#         self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)
#
#         self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)
#
#        # self.active = torch.nn.Sigmoid()
#
#
#     def forward(self, x):
#
#         e1 = self.RRCNN1(x)
#
#         e2 = self.Maxpool(e1)
#         e2 = self.RRCNN2(e2)
#
#         e3 = self.Maxpool1(e2)
#         e3 = self.RRCNN3(e3)
#
#         e4 = self.Maxpool2(e3)
#         e4 = self.RRCNN4(e4)
#
#         e5 = self.Maxpool3(e4)
#         e5 = self.RRCNN5(e5)
#
#         d5 = self.Up5(e5)
#         d5 = torch.cat((e4, d5), dim=1)
#         d5 = self.Up_RRCNN5(d5)
#
#         d4 = self.Up4(d5)
#         d4 = torch.cat((e3, d4), dim=1)
#         d4 = self.Up_RRCNN4(d4)
#
#         d3 = self.Up3(d4)
#         d3 = torch.cat((e2, d3), dim=1)
#         d3 = self.Up_RRCNN3(d3)
#
#         d2 = self.Up2(d3)
#         d2 = torch.cat((e1, d2), dim=1)
#         d2 = self.Up_RRCNN2(d2)
#
#         out = self.Conv(d2)
#
#       # out = self.active(out)
#
#         return out

class AttU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        out = self.active(out)

        return out

# class R2AttU_Net(nn.Module):
#     def __init__(self, in_ch=1, out_ch=1, t=2):
#         super(R2AttU_Net, self).__init__()
#
#         n1 = 64
#         filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
#
#         self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.RRCNN1 = RRCNN_block(in_ch, filters[0], t=t)
#         self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)
#         self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)
#         self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)
#         self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)
#
#         self.Up5 = up_conv(filters[4], filters[3])
#         self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
#         self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)
#
#         self.Up4 = up_conv(filters[3], filters[2])
#         self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
#         self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)
#
#         self.Up3 = up_conv(filters[2], filters[1])
#         self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
#         self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)
#
#         self.Up2 = up_conv(filters[1], filters[0])
#         self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
#         self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)
#
#         self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
#
#        # self.active = torch.nn.Sigmoid()
#
#
#     def forward(self, x):
#
#         e1 = self.RRCNN1(x)
#
#         e2 = self.Maxpool1(e1)
#         e2 = self.RRCNN2(e2)
#
#         e3 = self.Maxpool2(e2)
#         e3 = self.RRCNN3(e3)
#
#         e4 = self.Maxpool3(e3)
#         e4 = self.RRCNN4(e4)
#
#         e5 = self.Maxpool4(e4)
#         e5 = self.RRCNN5(e5)
#
#         d5 = self.Up5(e5)
#         e4 = self.Att5(g=d5, x=e4)
#         d5 = torch.cat((e4, d5), dim=1)
#         d5 = self.Up_RRCNN5(d5)
#
#         d4 = self.Up4(d5)
#         e3 = self.Att4(g=d4, x=e3)
#         d4 = torch.cat((e3, d4), dim=1)
#         d4 = self.Up_RRCNN4(d4)
#
#         d3 = self.Up3(d4)
#         e2 = self.Att3(g=d3, x=e2)
#         d3 = torch.cat((e2, d3), dim=1)
#         d3 = self.Up_RRCNN3(d3)
#
#         d2 = self.Up2(d3)
#         e1 = self.Att2(g=d2, x=e1)
#         d2 = torch.cat((e1, d2), dim=1)
#         d2 = self.Up_RRCNN2(d2)
#
#         out = self.Conv(d2)
#
#       #  out = self.active(out)
#
#         return out


# class NestedUNet(nn.Module):
#
#     def __init__(self, in_ch=1, out_ch=1):
#         super(NestedUNet, self).__init__()
#
#         n1 = 64
#         filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
#
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#
#         self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
#         self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
#         self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
#         self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
#         self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
#
#         self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
#         self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
#         self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
#         self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])
#
#         self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
#         self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
#         self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
#
#         self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
#         self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
#
#         self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
#
#         self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)
#
#     def forward(self, x):
#         x0_0 = self.conv0_0(x)
#         x1_0 = self.conv1_0(self.pool(x0_0))
#         x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))
#
#         x2_0 = self.conv2_0(self.pool(x1_0))
#         x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
#         x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))
#
#         x3_0 = self.conv3_0(self.pool(x2_0))
#         x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
#         x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
#         x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))
#
#         x4_0 = self.conv4_0(self.pool(x3_0))
#         x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
#         x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
#         x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
#         x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))
#
#         output = self.final(x0_4)
#         return output
