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
    layer1 keeps the same spatial size (stride = 1) and channels, while layer2, layer3 and layer4 progressively reduce spatial resolution (stride = 2)
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

    A standard ResNet18 computes this progression:
    stem   → 64 channels
    layer1 → 64 channels
    layer2 → 128 channels
    layer3 → 256 channels
    layer4 → 512 channels

    3. Classification head: 
    After layer4, the network has a compact feature map with many channels.
    Adaptive average pooling reduces each spatial feature map to a single value,
    converting the tensor from [B, 512, H, W] to [B, 512, 1, 1].
    The flatten operation then converts it to a vector [B, 512].
    Finally, the fully connected layer maps this feature vector to the desired
    number of output classes.

    Expected output for the classification version of ResNet-18:
    If the input tensor has shape [B, 3, H, W], the network outputs a tensor
    with shape [B, num_classes].
    For example, with B = 1 and num_classes = 10, the expected output is [1, 10].
    Each value is a raw class score, called a logit, not a probability. Es: output = [2.1, -0.4, 0.8, 3.2, ...]

    Note for HemoSet blood segmentation:
    The average pooling, flatten, and fully connected layers are part of the
    standard ResNet-18 classification head. They collapse the spatial feature map
    into a single vector and produce one label for the whole image.
    For HemoSet, this is not the final goal: we need semantic segmentation,
    meaning a pixel-level blood/background mask.
    Therefore, in the final U-Net model, these layers will be removed or ignored.
    We will keep only the ResNet encoder features from stem, layer1, layer2,
    layer3, and layer4, and pass them to a decoder that reconstructs the mask.
    """    

    def __init__(self, in_channels: int, out_channels: int, num_classes = 10):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, stride = 2, padding=1)
        )

        # Layer1 receives the stem output
        self.in_channels = out_channels

        # Layers: BasicBlock, out_channels, stride
        self.layer1 = self._make_layer(BasicBlock, 64, 2, 1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, 2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, 2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, 2)

        # Pooling and Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, basicblock, out_channels, num_basicblocks, stride):
        # In ResNet-18, each layer is composed of 2 BasicBlocks. 

        # Only the first basicblock of the layer may use stride > 1 to downsample
        # the spatial resolution. The remaining blocks use stride = 1.

        strides = [stride] + [1] * (num_basicblocks - 1)
        layers = []
        for stride in strides:
            layers.append(basicblock(self.in_channels, out_channels, stride))
            # After the first basicblock, the number of input channels for the next basicblock
            # becomes equal to out_channels.
            self.in_channels = out_channels

        # Combine all BasicBlocks into a single Sequential module.
        return nn.Sequential(*layers)
    

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out