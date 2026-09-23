import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SCDown(nn.Module):
    """ Spatial-Channel Decoupled Downsampling Block """
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k, s, g=c2, act=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))

class CIB(nn.Module):
    """ Compact Inverted Block """
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c1, 3, 1, g=c1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c2, 3, 1, g=c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))

class C2fCIB(nn.Module):
    """ C2f with Compact Inverted Bottleneck """
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([CIB(self.c, self.c, shortcut, e=1.0) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class PSA(nn.Module):
    """ Position-Sensitive Attention Block """
    def __init__(self, c, e=0.5):
        super().__init__()
        self.c = int(c * e)
        self.cv1 = Conv(c, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c, 1, 1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(self.c, self.c, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        a = a * self.attn(a)
        return self.cv2(torch.cat([a, b], 1))

class YOLOv10Backbone(nn.Module):
    """
    YOLOv10 Multi-Scale Lightweight Backbone & Neck Extractor.
    Extracts P3 (stride 8), P4 (stride 16), P5 (stride 32) features.
    Supports Pre-trained Weight Loading.
    """
    def __init__(self, in_channels=3, weights_path=None, **kwargs):
        super().__init__()
        # Backbone Layers
        self.p1 = Conv(in_channels, 64, 3, 2)
        self.p2 = nn.Sequential(Conv(64, 128, 3, 2), C2fCIB(128, 128, n=1))
        self.p3 = nn.Sequential(Conv(128, 256, 3, 2), C2fCIB(256, 256, n=2))
        self.p4 = nn.Sequential(SCDown(256, 512), C2fCIB(512, 512, n=2))
        self.p5 = nn.Sequential(SCDown(512, 1024), C2fCIB(1024, 1024, n=1), PSA(1024))
        self.out_channels = [256, 512, 1024]

        if weights_path:
            self.load_pretrained_weights(weights_path)

    def load_pretrained_weights(self, weights_path):
        """ Loads Pre-trained COCO/YOLOv10 weights safely """
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model'].float().state_dict()
            
            # Load only matching backbone keys
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Successfully loaded {len(pretrained_dict)} layers from {weights_path}")
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights from {weights_path}. Error: {e}")

    def forward(self, x):
        x1 = self.p1(x)
        x2 = self.p2(x1)
        p3 = self.p3(x2)  # Stride 8  [B, 256, H/8, W/8]
        p4 = self.p4(p3)  # Stride 16 [B, 512, H/16, W/16]
        p5 = self.p5(p4)  # Stride 32 [B, 1024, H/32, W/32]
        return [p3, p4, p5]

# Alias to support 'Mono3DBackbone' imports in Mono3DNetwork
Mono3DBackbone = YOLOv10Backbone