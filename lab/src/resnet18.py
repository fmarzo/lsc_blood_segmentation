import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Residual Block

    Structure
    ---------
    input x
        ↓ conv3x3_layer
        ↓ BatchNorm2d
        ↓ ReLU
        ↓ conv3x3_layer
        ↓ BatchNorm2d

    residual path:
        x oppure downsample(x)

    output:
        main path + residual path
        ↓ ReLU
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        #Computing F(x) 
        self.F = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        #Computing G(x): Residual Connection
        if in_channels != out_channels or stride != 1:
            self.G = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.G = nn.Identity()

        # ReLU finale dopo la somma F(x) + G(x)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.F(x)
        residual = self.G(x)

        out = out + residual
        out = self.relu(out)

        return out      


class ResNet18(nn.Module):

    """
    1.Stem
    Prepares the input tensor before passing it to the layers, we get the initial feature map. 

    Structure
    ---------
    input x
        ↓ conv7x7_layer
        ↓ BatchNorm2d
        ↓ ReLU
        ↓ MaxPool2d
    ---------

    2.Layers
    ResNet layers are groups of residual BasicBlocks.
    They take the feature map produced by the stem as input and progressively extract deeper features. 
    In ResNet-18, each layer contains 2 BasicBlocks:
    layer1 keeps the same spatial size and channels, while layer2, layer3 and layer4 progressively reduce spatial resolution 
    and increase channels. 
    Early layers capture simple low-level features, while deeper layers capture more complex high-level features.
    
    Structure
    ---------
    stem
    ↓
    layer1: BasicBlock + BasicBlock
    ↓
    layer2: BasicBlock + BasicBlock
    ↓
    layer3: BasicBlock + BasicBlock
    ↓
    layer4: BasicBlock + BasicBlock
    ---------

    """    

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

#       self.in_channels = in_channels
#       self.out_channels = out_channels

        #Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, stride = 2, padding=1)
        )

        #Layers: BasicBlock, out_channels, stride
        #self.layer1 = self.make_layer()

    def forward(self, x):
        out = self.stem(x)
        return out
