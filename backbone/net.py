#!/usr/bin/python
# -*- encoding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_norm import spectral_norm as SpectralNorm
import functools
import torch.nn.init as init
from torchsummary import summary


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


####################################################### Blocks #########################################################
class ResidualBlock(nn.Module):
    """Residual Block."""

    def __init__(self, dim_in, dim_out, net_mode=None):
        if net_mode == 'p' or (net_mode is None):
            use_affine = True
        elif net_mode == 't':
            use_affine = False
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine)
        )

    def forward(self, x):
        return x + self.main(x)


class GetMatrix(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GetMatrix, self).__init__()
        self.get_gamma = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.get_beta = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        gamma = self.get_gamma(x)
        beta = self.get_beta(x)
        return x, gamma, beta


class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class NONLocalBlock2D(nn.Module):
    def __init__(self):
        super(NONLocalBlock2D, self).__init__()
        self.g = nn.Conv2d(in_channels=1, out_channels=1,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, source, weight):
        """(b, c, h, w)
        src_diff: (3, 136, 32, 32)
        """
        batch_size = source.size(0)

        g_source = source.view(batch_size, 1, -1)  # (N, C, H*W)
        g_source = g_source.permute(0, 2, 1)  # (N, H*W, C)

        y = torch.bmm(weight.to_dense(), g_source)
        y = y.permute(0, 2, 1).contiguous()  # (N, C, H*W)
        y = y.view(batch_size, 1, *source.size()[2:])
        return y


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


# Makeup Transfer backbone: PSGAN https://github.com/wtjiang98/PSGAN
##################################################### Generator ########################################################
class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""

    def __init__(self):
        super(Generator, self).__init__()

        # -------------------------- PNet(MDNet) for obtaining makeup matrices --------------------------

        layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        self.pnet_in = layers

        # Down-Sampling
        curr_dim = 64
        for i in range(2):
            layers = nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.ReLU(inplace=True),
            )

            setattr(self, f'pnet_down_{i + 1}', layers)
            curr_dim = curr_dim * 2

        # Bottleneck. All bottlenecks share the same attention module
        self.atten_bottleneck_g = NONLocalBlock2D()
        self.atten_bottleneck_b = NONLocalBlock2D()
        self.simple_spade = GetMatrix(curr_dim, 1)  # get the makeup matrix

        for i in range(3):
            setattr(self, f'pnet_bottleneck_{i + 1}', ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='p'))

        # --------------------------- TNet(MANet) for applying makeup transfer ----------------------------

        self.tnet_in_conv = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.tnet_in_spade = nn.InstanceNorm2d(64, affine=False)
        self.tnet_in_relu = nn.ReLU(inplace=True)

        # Down-Sampling
        curr_dim = 64
        for i in range(2):
            setattr(self, f'tnet_down_conv_{i + 1}',
                    nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            setattr(self, f'tnet_down_spade_{i + 1}', nn.InstanceNorm2d(curr_dim * 2, affine=False))
            setattr(self, f'tnet_down_relu_{i + 1}', nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(6):
            setattr(self, f'tnet_bottleneck_{i + 1}', ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t'))

        # Up-Sampling
        for i in range(2):
            setattr(self, f'tnet_up_conv_{i + 1}',
                    nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            setattr(self, f'tnet_up_spade_{i + 1}', nn.InstanceNorm2d(curr_dim // 2, affine=False))
            setattr(self, f'tnet_up_relu_{i + 1}', nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers = nn.Sequential(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )
        self.tnet_out = layers

    @staticmethod
    def atten_feature(mask_s, weight, gamma_s, beta_s, atten_module_g, atten_module_b):
        """
        feature size: (1, c, h, w)
        mask_c(s): (3, 1, h, w)
        diff_c: (1, 138, 256, 256)
        return: (1, c, h, w)
        """
        channel_num = gamma_s.shape[1]

        mask_s_re = F.interpolate(mask_s, size=gamma_s.shape[2:]).repeat(1, channel_num, 1, 1)
        gamma_s_re = gamma_s.repeat(3, 1, 1, 1)
        gamma_s = gamma_s_re * mask_s_re  # (3, c, h, w)
        beta_s_re = beta_s.repeat(3, 1, 1, 1)
        beta_s = beta_s_re * mask_s_re

        gamma = atten_module_g(gamma_s, weight)  # (3, c, h, w)
        beta = atten_module_b(beta_s, weight)

        gamma = (gamma[0] + gamma[1] + gamma[2]).unsqueeze(0)  # (c, h, w) combine the three parts
        beta = (beta[0] + beta[1] + beta[2]).unsqueeze(0)
        return gamma, beta

    def get_weight(self, mask_c, mask_s, fea_c, fea_s, diff_c, diff_s):
        """  s --> source; c --> target
        feature size: (1, 256, 64, 64)
        diff: (3, 136, 32, 32)
        """
        HW = 64 * 64
        batch_size = 3
        assert fea_s is not None  # fea_s when i==3
        # get 3 part fea using mask
        channel_num = fea_s.shape[1]

        mask_c_re = F.interpolate(mask_c, size=64).repeat(1, channel_num, 1, 1)  # (3, c, h, w)
        fea_c = fea_c.repeat(3, 1, 1, 1)  # (3, c, h, w)
        fea_c = fea_c * mask_c_re  # (3, c, h, w) 3 stands for 3 parts

        mask_s_re = F.interpolate(mask_s, size=64).repeat(1, channel_num, 1, 1)
        fea_s = fea_s.repeat(3, 1, 1, 1)
        fea_s = fea_s * mask_s_re

        theta_input = torch.cat((fea_c * 0.01, diff_c), dim=1)
        phi_input = torch.cat((fea_s * 0.01, diff_s), dim=1)

        theta_target = theta_input.view(batch_size, -1, HW)  # (N, C+136, H*W)
        theta_target = theta_target.permute(0, 2, 1)  # (N, H*W, C+136)
        phi_source = phi_input.view(batch_size, -1, HW)  # (N, C+136, H*W)
        weight = torch.bmm(theta_target, phi_source)  # (3, HW, HW)
        with torch.no_grad():
            v = weight.detach().nonzero().long().permute(1, 0)
            # This clone is required to correctly release cuda memory.
            weight_ind = v.clone()
            del v
            torch.cuda.empty_cache()

        weight *= 200  # hyper parameters for visual feature
        weight = F.softmax(weight, dim=-1)
        weight = weight[weight_ind[0], weight_ind[1], weight_ind[2]]
        ret = torch.sparse.FloatTensor(weight_ind, weight, torch.Size([3, HW, HW]))
        return ret

    def forward(self, c, s, mask_c, mask_s, diff_c, diff_s, gamma=None, beta=None, ret=False):
        c, s, mask_c, mask_s, diff_c, diff_s = [x.squeeze(0) if x.ndim == 5 else x for x in
                                                [c, s, mask_c, mask_s, diff_c, diff_s]]
        """attention version
        c: content, stands for source image. shape: (b, c, h, w)
        s: style, stands for reference image. shape: (b, c, h, w)
        mask_list_c: lip, skin, eye. (b, 1, h, w)
        """

        # forward c in tnet(MANet)
        c_tnet = self.tnet_in_conv(c)
        s = self.pnet_in(s)
        c_tnet = self.tnet_in_spade(c_tnet)
        c_tnet = self.tnet_in_relu(c_tnet)

        # down-sampling
        for i in range(2):
            if gamma is None:
                cur_pnet_down = getattr(self, f'pnet_down_{i + 1}')
                s = cur_pnet_down(s)

            cur_tnet_down_conv = getattr(self, f'tnet_down_conv_{i + 1}')
            cur_tnet_down_spade = getattr(self, f'tnet_down_spade_{i + 1}')
            cur_tnet_down_relu = getattr(self, f'tnet_down_relu_{i + 1}')
            c_tnet = cur_tnet_down_conv(c_tnet)
            c_tnet = cur_tnet_down_spade(c_tnet)
            c_tnet = cur_tnet_down_relu(c_tnet)

        # bottleneck
        for i in range(6):
            if gamma is None and i <= 2:
                cur_pnet_bottleneck = getattr(self, f'pnet_bottleneck_{i + 1}')
            cur_tnet_bottleneck = getattr(self, f'tnet_bottleneck_{i + 1}')

            # get s_pnet from p and transform
            if i == 3:
                if gamma is None:  # not in test_mix
                    s, gamma, beta = self.simple_spade(s)
                    weight = self.get_weight(mask_c, mask_s, c_tnet, s, diff_c, diff_s)
                    gamma, beta = self.atten_feature(mask_s, weight, gamma, beta, self.atten_bottleneck_g,
                                                     self.atten_bottleneck_b)
                    if ret:
                        return [gamma, beta]
                # else:                       # in test mode
                # gamma, beta = param_A[0]*w + param_B[0]*(1-w), param_A[1]*w + param_B[1]*(1-w)

                c_tnet = c_tnet * (1 + gamma) + beta  # apply makeup transfer using makeup matrices

            if gamma is None and i <= 2:
                s = cur_pnet_bottleneck(s)
            c_tnet = cur_tnet_bottleneck(c_tnet)

        # up-sampling
        for i in range(2):
            cur_tnet_up_conv = getattr(self, f'tnet_up_conv_{i + 1}')
            cur_tnet_up_spade = getattr(self, f'tnet_up_spade_{i + 1}')
            cur_tnet_up_relu = getattr(self, f'tnet_up_relu_{i + 1}')
            c_tnet = cur_tnet_up_conv(c_tnet)
            c_tnet = cur_tnet_up_spade(c_tnet)
            c_tnet = cur_tnet_up_relu(c_tnet)

        c_tnet = self.tnet_out(c_tnet)
        return c_tnet


################################################### Discriminator ######################################################
class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, repeat_num=3, norm='SN'):
        super(Discriminator, self).__init__()

        layers = []
        if norm == 'SN':
            layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        else:
            layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            if norm == 'SN':
                layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
            else:
                layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        # k_size = int(image_size / np.power(2, repeat_num))
        if norm == 'SN':
            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=1, padding=1)))
        else:
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        if norm == 'SN':
            self.conv1 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False)

        # conv1 remain the last square size, 256*256-->30*30
        # self.conv2 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=k_size, bias=False))
        # conv2 output a single number

    def forward(self, x):
        if x.ndim == 5:
            x = x.squeeze(0)
        assert x.ndim == 4, x.ndim
        h = self.main(x)
        # out_real = self.conv1(h)
        out_makeup = self.conv1(h)
        # return out_real.squeeze(), out_makeup.squeeze()
        return out_makeup

#################################################### Regularizer #######################################################
class H_RRDB(nn.Module):
    def __init__(self, in_nc=3, nf=64, nb=23, gc=32):
        super(H_RRDB, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        return trunk


###################################################### for test ########################################################
def weights_init_xavier(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(model.weight.data, gain=1.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal(model.weight.data, gain=1.0)


def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))


if __name__ == '__main__':
    # test
    device = 'cuda'
    test_net = H_RRDB(nb=5).to(device)
    weights_init_xavier(test_net)
    test_sample = torch.rand(1, 3, 512, 512).to(device)
    print_network(test_net)
    summary(test_net, input_size=[(3, 512, 512)], batch_size=1, device='cuda')
    print(test_sample.size())
    print(test_net(test_sample).size())
