import torch.nn as nn
import torch

class SingleConv_torch(nn.Module):
    def __init__(self) -> None:
        super(SingleConv_torch , self).__init__()
        self.backbone_1 = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=32 , kernel_size=5 , 
                        stride=1 , padding=0)
        )
    
    def forward(self , x):
        input_ = x
        temp_1 = self.backbone_1(input_)
        return temp_1