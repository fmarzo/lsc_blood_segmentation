from src.resnet18 import BasicBlock
from src.resnet18 import ResNet18
import torch

#TEST RESIDUAL BLOCK

# #esempio 1: No downsampling in residual connection, output atteso: n × 64 × 56 × 56

# x = torch.rand(1, 64, 56, 56)

# block = BasicBlock(
#     in_channels=x.shape[1],
#     out_channels=64,
#     stride=1
# )

# out = block(x) #pytorch passa attraverso logica di nn.module e chiama il metodo forward in basicblock

# print(out.shape)

# #esempio 2: downsampling in residual connection, ouput atteso: n 128 × 28 × 28

# x = torch.rand(1, 64, 56, 56)

# block = BasicBlock(
#     in_channels=x.shape[1],
#     out_channels=128,
#     stride=2
# )

# out = block(x) #pytorch passa attraverso logica di nn.module e chiama il metodo forward in basicblock

# print(out.shape)

#--------------------------------------------------------------------------------

#TEST RESNET18 (Stem+Layer)

x = torch.rand(1, 3, 224, 224)

model = ResNet18(
    in_channels=3,
    out_channels=64,
    num_classes=10
)

out = model(x)

print(out.shape)

#----------------------------------------------------------------------------------

