import numpy as np
import torch
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
import torch.nn.init as init

from model.helper import get_blocks, bottleneck_IR, bottleneck_IR_SE
from stylegan2.model import EqualLinear, Generator

class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', in_dim=3):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(in_dim, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = 18
        self.coarse_ind = 3
        self.middle_ind = 7
        self.styles.append(GradualStyleBlock(512, 512, 16))
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        w0 = self.styles[0](c3)
        w = w0
        return w

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #init.constant_(m.weight.data, 0)
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
            #init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)

class pSp(nn.Module):

    def __init__(self, in_dim=3, enc_ckpt=None, gan_ckpt = None):
        super(pSp, self).__init__()
        # Define architecture
        self.encoder = GradualStyleEncoder(50, 'ir_se', in_dim=in_dim)
        self.decoder = Generator(1024, 512, 8)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights(enc_ckpt, gan_ckpt)

    def load_weights(self, enc_ckpt, gan_ckpt):
        print('Loading encoder weights from ckpt!')
        ckpt = torch.load(enc_ckpt)
        self.encoder.load_state_dict(ckpt, strict=True)

        print('Loading decoder weights from pretrained!')
        ckpt = torch.load(gan_ckpt)
        self.decoder.load_state_dict(ckpt['g_ema'], strict=True)
        del ckpt

    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):

        codes = x

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code
        images, result_latent = self.decoder([codes],
                                             input_is_latent=True,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images


class Mapper(Module):
    def __init__(self, in_dim):
        super(Mapper, self).__init__()

        layers = [
            nn.Linear(in_dim+512, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
        ]
        
        self.style = nn.Sequential(*layers)

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                weights_init(m)

    def forward(self, x):
        x = self.style(x)
        return x

class condi(nn.Module):

    def __init__(self, in_dim=100):
        super(condi, self).__init__()

        self.style1 = Mapper(in_dim)
        self.style2 = Mapper(in_dim)
        self.style3 = Mapper(in_dim)

    def forward(self, w, age, gender):
        xx = torch.cat((w, age.repeat(18, 1, 50).permute(1, 0, 2), gender.repeat(18, 1, 50).permute(1, 0, 2)), 2)
        
        x_coarse = xx[:, :4, :]
        x_medium = xx[:, 4:8, :]
        x_fine = xx[:, 8:, :]

        x_coarse = self.style1(x_coarse)
        x_medium = self.style2(x_medium)
        x_fine = self.style3(x_fine)

        return torch.cat([x_coarse, x_medium, x_fine], dim=1)

