import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .video_net import GDN, flow_warp, ResBlock, ResBlock_LeakyReLU_0_Point_1
from ..entropy_models.video_entropy_models import BitEstimator, GaussianEncoder
from ..layers.layers import MaskedConv2d, subpel_conv3x3
from .Restormer import TransformerBlock 

def save_model(model, iter):
    torch.save(model.state_dict(), "./snapshot/iter{}.model".format(iter))


def load_model(model, f):
    print("load DCVC format")
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        result_dict = {}
        i = 0
        v = list(pretrained_dict.values())
        for key, weight in model_dict.items():
            result_key = key
            result_dict[result_key] = v[i]
            i += 1
        model_dict.update(result_dict)
        model.load_state_dict(model_dict)

    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])    #return step
    else:
        return 0

class B_DCVC_net(nn.Module):
    def __init__(self):
        super().__init__()
        out_channel_N = 64
        out_channel_M = 96
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

        self.bitEstimator_z = BitEstimator(out_channel_N)

        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
        )

        self.context_refine = nn.Sequential(
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),
        )

        self.gaussian_encoder = GaussianEncoder()
        self.contextualEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N+3, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),
        )

        self.contextualDecoder_part1 = ContextualDecoder_net_part1()
        self.contextualDecoder_part2 = nn.Sequential(
            nn.Conv2d(out_channel_N*3, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
            *[TransformerBlock(dim=out_channel_N, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(4)],
            nn.Conv2d(out_channel_N, 3, 3, stride=1, padding=1),
        )

        self.priorEncoder = PriorEncoder_net(out_channel_M, out_channel_N)
        self.priorDecoder = PriorDecoder_net(in_channel=out_channel_N, out_channel=out_channel_M)

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(out_channel_M * 12 // 3, out_channel_M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 10 // 3, out_channel_M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 8 // 3, out_channel_M * 6 // 3, 1),
        )

        self.auto_regressive = MaskedConv2d(
            out_channel_M, 2 * out_channel_M, kernel_size=5, padding=2, stride=1
        )

        self.temporalPriorEncoder = TemporalPriorEncoder_net(out_channel_N, out_channel_M)
        self.moduleBlend = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid()
        )

    def feature_probs_based_sigma(self, feature, mean, sigma):
        outputs = feature
        values = outputs - mean
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(values + 0.5) - gaussian.cdf(values - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, probs

    def iclr18_estrate_bits_z(self, z):
        prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def forward(self, input_image, referframe, quant_noise_feature=None, quant_noise_z=None):

        alpha = self.moduleBlend(torch.abs(input_image-referframe))
        input_image_alpha = input_image * alpha

        ref_feature = self.feature_extract(referframe)
        context = self.context_refine(ref_feature)

        temporal_prior_params = self.temporalPriorEncoder(context)

        feature = self.contextualEncoder(torch.cat((input_image_alpha, context), dim=1))
        z = self.priorEncoder(feature)
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)
        
        params = self.priorDecoder(compressed_z)

        feature_renorm = feature

        if self.training:
            compressed_y_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_y_renorm = torch.round(feature_renorm)

        ctx_params = self.auto_regressive(compressed_y_renorm)
        gaussian_params = self.entropy_parameters(
            torch.cat((temporal_prior_params, params, ctx_params), dim=1)
        )

        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        recon_image_feature = self.contextualDecoder_part1(compressed_y_renorm)
        recon_image_feature_refine = self.mv_refine(referframe, recon_image_feature)
        recon_image = self.contextualDecoder_part2(torch.cat((recon_image_feature_refine, recon_image_feature, context) , dim=1))

        clipped_recon_image = recon_image.clamp(0., 1.)

        total_bits_y, _ = self.feature_probs_based_sigma(compressed_y_renorm, means_hat, scales_hat)
        total_bits_z, _ = self.iclr18_estrate_bits_z(compressed_z)

        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_y = total_bits_y / pixel_num
        bpp_z = total_bits_z / pixel_num
        bpp = bpp_y + bpp_z 

        # distortion
        distortion1 = torch.mean((referframe - input_image).pow(2))
        distortion2 = torch.mean((recon_image - input_image).pow(2))

        return clipped_recon_image, distortion1, distortion2, bpp_y, bpp_z, bpp

# =============================================================================================
# =============================================================================================
# =============================================================================================

class PriorEncoder_net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PriorEncoder_net, self).__init__()
        self.l1 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.l1.weight.data, (math.sqrt(2 * (in_channel + out_channel) / (in_channel + in_channel))))
        torch.nn.init.constant_(self.l1.bias.data, 0.01)
        self.r1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l2.bias.data, 0.01)
        self.r2 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l3.bias.data, 0.01)
    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.l3(x)
        return x

class PriorDecoder_net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PriorDecoder_net, self).__init__()
        self.l1 = nn.ConvTranspose2d(in_channel, out_channel, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.l1.weight.data, (math.sqrt(2 * (in_channel + out_channel) / (in_channel + in_channel))))
        torch.nn.init.constant_(self.l1.bias.data, 0.01)
        self.r1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.ConvTranspose2d(out_channel, out_channel, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.l2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l2.bias.data, 0.01)
        self.r2 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.ConvTranspose2d(out_channel, out_channel, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.l3.weight.data, (math.sqrt(2 * (in_channel + out_channel) / (out_channel + out_channel))))
        torch.nn.init.constant_(self.l3.bias.data, 0.01)
    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.l3(x)
        return x

class Entropy_parameters_net(nn.Module):
    def __init__(self, channel):
        super(Entropy_parameters_net, self).__init__()
        self.l1 = nn.Conv2d(channel, channel, 1)
        torch.nn.init.xavier_normal_(self.l1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l1.bias.data, 0.01)
        self.r1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.Conv2d(channel, channel, 1)
        torch.nn.init.xavier_normal_(self.l2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l2.bias.data, 0.01)
        self.r2 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.Conv2d(channel, channel, 1)
        torch.nn.init.xavier_normal_(self.l3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l3.bias.data, 0.01)
    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.l3(x)
        return x

class TemporalPriorEncoder_net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TemporalPriorEncoder_net, self).__init__()
        self.l1 = nn.Conv2d(in_channel, in_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l1.bias.data, 0.01)
        self.r1 = GDN(in_channel)
        self.l2 = nn.Conv2d(in_channel, in_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l2.bias.data, 0.01)
        self.r2 = GDN(in_channel)
        self.l3 = nn.Conv2d(in_channel, in_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l3.bias.data, 0.01)
        self.r3 = GDN(in_channel)
        self.l4 = nn.Conv2d(in_channel, out_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l4.weight.data, (math.sqrt(2*(in_channel+out_channel)/(2*in_channel))))
        torch.nn.init.constant_(self.l4.bias.data, 0.01)
    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.r3(self.l3(x))
        x = self.l4(x)
        return x


out_channel_N = 64
out_channel_M = 96

# =============================================================================================
# =============================================================================================
# =============================================================================================

class ContextualDecoder_net_part1(nn.Module):
    '''
    Contextual Decoder part1
    '''
    def __init__(self):
        super(ContextualDecoder_net_part1, self).__init__()
        self.deconv1 = subpel_conv3x3(out_channel_M, out_channel_N, 2)
        self.gdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = subpel_conv3x3(out_channel_N, out_channel_N, 2)
        self.gdn2 = GDN(out_channel_N, inverse=True)
        self.res1 = ResBlock_LeakyReLU_0_Point_1(out_channel_N)
        self.deconv3 = subpel_conv3x3(out_channel_N, out_channel_N, 2)
        self.gdn3 = GDN(out_channel_N, inverse=True)
        self.res2 = ResBlock_LeakyReLU_0_Point_1(out_channel_N)
        self.deconv4 = subpel_conv3x3(out_channel_N, out_channel_N, 2)

        self.context_synthesis = GridNet(out_channel_N+60, out_channel_N)

        self.module1by1_1 = torch.nn.Conv2d(in_channels=out_channel_M, out_channels=4, kernel_size=1, stride=1, padding=1)
        self.module1by1_2 = torch.nn.Conv2d(in_channels=out_channel_N, out_channels=8, kernel_size=1, stride=1, padding=1)
        self.module1by1_3 = torch.nn.Conv2d(in_channels=out_channel_N, out_channels=12, kernel_size=1, stride=1, padding=1)
        self.module1by1_4 = torch.nn.Conv2d(in_channels=out_channel_N, out_channels=16, kernel_size=1, stride=1, padding=1)
        self.module1by1_5 = torch.nn.Conv2d(in_channels=out_channel_N, out_channels=20, kernel_size=1, stride=1, padding=1)

        self.moduleBlend = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_channel_N, out_channels=out_channel_N // 2, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=out_channel_N // 2, out_channels=out_channel_N // 4, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=out_channel_N // 4, out_channels=out_channel_N // 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=out_channel_N // 8, out_channels=1, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x1):
        x2 = self.gdn1(self.deconv1(x1))
        x3 = self.res1(self.gdn2(self.deconv2(x2)))
        x4 = self.res2(self.gdn3(self.deconv3(x3)))
        x5 = self.deconv4(x4)
        
        featConv1 = self.module1by1_1(x1)
        featConv2 = self.module1by1_2(x2)
        featConv3 = self.module1by1_3(x3)
        featConv4 = self.module1by1_4(x4)
        featConv5 = self.module1by1_5(x5)

        w, h = x5.shape[2:]
        tensorConv1 = F.interpolate(featConv1, size=(w, h), mode='bilinear', align_corners=False)
        tensorConv2 = F.interpolate(featConv2, size=(w, h), mode='bilinear', align_corners=False)
        tensorConv3 = F.interpolate(featConv3, size=(w, h), mode='bilinear', align_corners=False)
        tensorConv4 = F.interpolate(featConv4, size=(w, h), mode='bilinear', align_corners=False)
        tensorConv5 = F.interpolate(featConv5, size=(w, h), mode='bilinear', align_corners=False)

        tensorCombined = torch.cat([x5, tensorConv1, tensorConv2, tensorConv3, tensorConv4, tensorConv5], dim=1)
        feature_feat = self.context_synthesis(tensorCombined)
        Blend = self.moduleBlend(x4)
        output = Blend * feature_feat + (1 - Blend) * x5
        
        return output


class GridNet(nn.Module):
    def __init__(self, in_chs, out_chs, grid_chs=(32, 64, 96)):
        super(GridNet, self).__init__()

        self.n_row = 3
        self.n_col = 6
        self.n_chs = grid_chs
        assert len(grid_chs) == self.n_row, 'should give num channels for each row (scale stream)'

        self.lateral_init = LateralBlock(in_chs, self.n_chs[0])

        for r, n_ch in enumerate(self.n_chs):
            for c in range(self.n_col - 1):
                setattr(self, f'lateral_{r}_{c}', LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'down_{r}_{c}', DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'up_{r}_{c}', UpSamplingBlock(in_ch, out_ch))

        self.lateral_final = LateralBlock(self.n_chs[0], out_chs)

    def forward(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)
        state_20 = self.down_1_0(state_10)

        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
        state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

        state_02 = self.lateral_0_1(state_01)
        state_12 = self.down_0_2(state_02) + self.lateral_1_1(state_11)
        state_22 = self.down_1_2(state_12) + self.lateral_2_1(state_21)

        state_23 = self.lateral_2_2(state_22)
        state_13 = self.up_1_0(state_23) + self.lateral_1_2(state_12)
        state_03 = self.up_0_0(state_13) + self.lateral_0_2(state_02)

        state_24 = self.lateral_2_3(state_23)
        state_14 = self.up_1_1(state_24) + self.lateral_1_3(state_13)
        state_04 = self.up_0_1(state_14) + self.lateral_0_3(state_03)

        state_25 = self.lateral_2_4(state_24)
        state_15 = self.up_1_2(state_25) + self.lateral_1_4(state_14)
        state_05 = self.up_0_2(state_15) + self.lateral_0_4(state_04)

        return self.lateral_final(state_05)

class LateralBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(LateralBlock, self).__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        )
        if ch_in != ch_out:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

    def forward(self, x):
        fx = self.f(x)
        if fx.shape[1] != x.shape[1]:
            x = self.conv(x)
        return fx + x

class DownSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DownSamplingBlock, self).__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.f(x)


class UpSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpSamplingBlock, self).__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.f(x)