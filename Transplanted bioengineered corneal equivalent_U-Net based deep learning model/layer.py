import torch
import torch.nn as nn


# ###################################################
# ###################################################
# ##################### UNet ########################
# ###################################################
# ###################################################
# class CBR2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
#         super().__init__()
#         ## Define layers
#         layers = []
#         ## Convolution layer
#         layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                              kernel_size=kernel_size, stride=stride, padding=padding,
#                              bias=bias)]
#         ## Batch normalization layer
#         if not norm is None:
#             if norm == "bnorm":
#                 layers += [nn.BatchNorm2d(num_features=out_channels)]
#             elif norm =="inorm":
#                 layers += [nn.InstanceNorm2d(num_features=out_channels)]
#
#         ## ReLu layer
#         if not relu is None:
#             layers += [nn.ReLU() if relu == 0.0 else nn.LeakyReLU(relu)]
#
#
#         self.cbr = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.cbr(x)
#
#
# class CBPR2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm"):
#         super().__init__()
#         ## Define layers
#         layers = []
#         ## Convolution layer
#         layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                              kernel_size=kernel_size, stride=stride, padding=padding,
#                              bias=bias)]
#         ## Batch normalization layer
#         if not norm is None:
#             if norm == "bnorm":
#                 layers += [nn.BatchNorm2d(num_features=out_channels)]
#             elif norm == "inorm":
#                 layers += [nn.InstanceNorm2d(num_features=out_channels)]
#
#         ## ReLu layer
#
#         layers += [nn.PReLU]
#
#         self.cbpr = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.cbpr(x)
#
# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels,
#                  kernel_size=3, stride=1, padding=1,
#                  bias=True, norm="bnorm", relu=0.0):
#         super().__init__()
#
#         layer = []
#
#         ## 1st CBR2d
#         layer += [CBR2d(in_channels, out_channels,
#                         kernel_size=kernel_size, stride=stride,
#                         padding=padding, bias=bias,
#                         norm=norm, relu=relu)]
#
#         ## 2nd CBR2d
#         layer += [CBR2d(in_channels, out_channels,
#                         kernel_size=kernel_size, stride=stride,
#                         padding=padding, bias=bias,
#                         norm=norm, relu=None)]
#
#         self.resblk = nn.Sequential(*layer)
#
#     def forward(self, x):
#         return x + self.resblk(x)





###################################################
###################################################
################## ResUNet ########################
###################################################
###################################################

# class ResidualConv(nn.Module):
#     def __init__(self, input_dim, output_dim, stride, padding):
#         super(ResidualConv, self).__init__()
#
#         self.conv_block = nn.Sequential(
#             nn.BatchNorm2d(input_dim),
#             nn.PReLU(),
#             nn.Conv2d(
#                 input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
#             ),
#             nn.BatchNorm2d(output_dim),
#             nn.PReLU(),
#             nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
#         )
#         self.conv_skip = nn.Sequential(
#             nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
#             nn.BatchNorm2d(output_dim),
#         )
#
#     def forward(self, x):
#
#         return self.conv_block(x) + self.conv_skip(x)
#
#
# class Upsample(nn.Module):
#     def __init__(self, input_dim, output_dim, kernel, stride):
#         super(Upsample, self).__init__()
#
#         self.upsample = nn.ConvTranspose2d(
#             input_dim, output_dim, kernel_size=kernel, stride=stride
#         )
#
#     def forward(self, x):
#         return self.upsample(x)
#
#
# class Squeeze_Excite_Block(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(Squeeze_Excite_Block, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.PReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)
#
#
# class ASPP(nn.Module):
#     def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
#         super(ASPP, self).__init__()
#
#         self.aspp_block1 = nn.Sequential(
#             nn.Conv2d(
#                 in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
#             ),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(out_dims),
#         )
#         self.aspp_block2 = nn.Sequential(
#             nn.Conv2d(
#                 in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
#             ),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(out_dims),
#         )
#         self.aspp_block3 = nn.Sequential(
#             nn.Conv2d(
#                 in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
#             ),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(out_dims),
#         )
#
#         self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
#         self._init_weights()
#
#     def forward(self, x):
#         x1 = self.aspp_block1(x)
#         x2 = self.aspp_block2(x)
#         x3 = self.aspp_block3(x)
#         out = torch.cat([x1, x2, x3], dim=1)
#         return self.output(out)
#
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#
# class Upsample_(nn.Module):
#     def __init__(self, scale=2):
#         super(Upsample_, self).__init__()
#
#         self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)
#
#     def forward(self, x):
#         return self.upsample(x)
#
#
# class AttentionBlock(nn.Module):
#     def __init__(self, input_encoder, input_decoder, output_dim):
#         super(AttentionBlock, self).__init__()
#
#         self.conv_encoder = nn.Sequential(
#             nn.BatchNorm2d(input_encoder),
#             nn.ReLU(),
#             nn.Conv2d(input_encoder, output_dim, 3, padding=1),
#             nn.MaxPool2d(2, 2),
#         )
#
#         self.conv_decoder = nn.Sequential(
#             nn.BatchNorm2d(input_decoder),
#             nn.ReLU(),
#             nn.Conv2d(input_decoder, output_dim, 3, padding=1),
#         )
#
#         self.conv_attn = nn.Sequential(
#             nn.BatchNorm2d(output_dim),
#             nn.ReLU(),
#             nn.Conv2d(output_dim, 1, 1),
#         )
#
#     def forward(self, x1, x2):
#         out = self.conv_encoder(x1) + self.conv_decoder(x2)
#         out = self.conv_attn(out)
#         return out * x2


##################################################
##################################################
################# RCNN UNet ######################
##################################################
##################################################
class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):

    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out


class RRCNN_block(nn.Module):

    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out

class Attention_block(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output