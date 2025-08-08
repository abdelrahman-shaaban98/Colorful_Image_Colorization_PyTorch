import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=True):
        super().__init__()
        
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                            padding=padding, dilation=dilation)]
        layers.append(nn.ReLU(inplace=True))
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    

class ColorizationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            ConvBlock(1, 64, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False),
            ConvBlock(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, batch_norm=True)
        )

        self.conv2 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False),
            ConvBlock(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, batch_norm=True)
        )

        self.conv3 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False),
            ConvBlock(256, 256, kernel_size=3, stride=2, padding=1, dilation=1, batch_norm=True)
        )

        self.conv4 = nn.Sequential(
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=True)
        )

        self.conv5 = nn.Sequential(
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=False),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=False),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=True)
        )

        self.conv6 = nn.Sequential(
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=False),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=False),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=True)
        )

        self.conv7 = nn.Sequential(
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=True)
        )

        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False)
        )
        
        self.model_out = nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, dilation=1)
        

    def forward(self, x):
        # Input x shape: [B, 1, 256, 256]
        x = self.conv1(x)    # -> [B, 64, 128, 128]
        x = self.conv2(x)    # -> [B, 128, 64, 64]
        x = self.conv3(x)    # -> [B, 256, 32, 32]
        x = self.conv4(x)    # -> [B, 512, 32, 32]
        x = self.conv5(x)    # -> [B, 512, 32, 32]
        x = self.conv6(x)    # -> [B, 512, 32, 32]
        x = self.conv7(x)    # -> [B, 512, 32, 32]
        x = self.conv8(x)    # -> [B, 256, 64, 64]
        out = self.model_out(x)    # -> [B, 313, 64, 64]

        return out
    
if __name__ == "__main__":
    model = ColorizationNet()
    t = torch.rand(8, 1, 256, 256)
    out = model(t)
    print(t.shape)
    print(out.shape)