"""CSRNet (Li, Zhang & Chen, CVPR 2018) -- VGG16 first-10-layer frontend +
6 dilated-conv backend, faithfully following the architecture from
leeyeehoo/CSRNet-pytorch, but corrected for Python 3 (the original repo's
weight-copying loop uses Python 2's `xrange` and indexes `.items()` like a
list, both invalid in Python 3 -- fixed here to use `load_state_dict`
directly instead of manually walking parameter tensors).
"""
import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super().__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            self._initialize_weights()
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            # vgg16.features has the same first-10-conv-layer structure as our
            # frontend (same channel progression); copy those pretrained
            # weights in directly via state_dict, matching by position.
            vgg_state = list(vgg16.features.state_dict().items())
            frontend_state = self.frontend.state_dict()
            frontend_keys = list(frontend_state.keys())
            for i, key in enumerate(frontend_keys):
                frontend_state[key] = vgg_state[i][1]
            self.frontend.load_state_dict(frontend_state)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == "__main__":
    m = CSRNet()
    x = torch.randn(1, 3, 768, 768)
    y = m(x)
    print("output shape:", y.shape, "(expect 1/8 spatial resolution, 1 channel)")
    print("predicted count (sum of density map):", y.sum().item())
