import torch
import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.up(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return skip_connection * psi

# 🔥 ASPP 모듈 정의
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        ] + [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            for r in rates[1:]
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * len(rates), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        res = torch.cat([stage(x) for stage in self.stages], dim=1)
        return self.bottleneck(res)

class AttentionUNet_MobileNetV3(nn.Module):
    def __init__(self, output_ch=4, pretrained=True):
        super(AttentionUNet_MobileNetV3, self).__init__()
        mobilenet = models.mobilenet_v3_large(weights='DEFAULT' if pretrained else None).features
        
        self.enc1 = mobilenet[0:2]    # 16 ch
        self.enc2 = mobilenet[2:4]    # 24 ch
        self.enc3 = mobilenet[4:7]    # 40 ch
        self.enc4 = mobilenet[7:13]   # 112 ch
        self.enc5 = mobilenet[13:16]  # 160 ch

        # 🔥 Bottleneck에 ASPP 배치
        self.aspp = ASPP(160, 160)

        self.Up5 = UpConv(160, 112)
        self.Att5 = AttentionBlock(F_g=112, F_l=112, n_coefficients=56)
        self.UpConv5 = ConvBlock(224, 112)

        self.Up4 = UpConv(112, 40)
        self.Att4 = AttentionBlock(F_g=40, F_l=40, n_coefficients=20)
        self.UpConv4 = ConvBlock(80, 40)

        self.Up3 = UpConv(40, 24)
        self.Att3 = AttentionBlock(F_g=24, F_l=24, n_coefficients=12)
        self.UpConv3 = ConvBlock(48, 24)

        self.Up2 = UpConv(24, 16)
        self.Att2 = AttentionBlock(F_g=16, F_l=16, n_coefficients=8)
        self.UpConv2 = ConvBlock(32, 16)

        self.Up_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Conv_1x1 = nn.Conv2d(16, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3); e5 = self.enc5(e4)
        
        # 🔥 ASPP 적용
        e5 = self.aspp(e5)

        d5 = self.Up5(e5)
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1); d5 = self.UpConv5(d5)

        d4 = self.Up4(d5); s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1); d4 = self.UpConv4(d4)

        d3 = self.Up3(d4); s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1); d3 = self.UpConv3(d3)

        d2 = self.Up2(d3); s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1); d2 = self.UpConv2(d2)

        out = self.Up_final(d2); out = self.Conv_1x1(out)
        return out