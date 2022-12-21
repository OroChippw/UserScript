import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 

from ConvTensorFlow import ConvTensorFlow


class ResModule(nn.Module):
    def __init__(self , in_channels_ , out_channels_ , stride_ = 1) -> None:
        super(ResModule , self).__init__()
        self.stride_ = stride_
        self.channelpad_ = out_channels_ - in_channels_
        kernel_size_ = 5
        if self.stride_ > 1 :
            self.max_pool_ = nn.MaxPool2d(
                kernel_size=[2,2] , stride=[stride_ , stride_] , padding=[0,0] , 
                dilation=1 , ceil_mode=False)
        self.convs_ = nn.Sequential(
            ConvTensorFlow(in_channels_=in_channels_ , out_channels_=in_channels_ , 
                            kernel_size_=kernel_size_ , stride_=stride_ , padding_='same' , 
                            groups_=in_channels_ , bias_=True),
            ConvTensorFlow(in_channels_=in_channels_ , out_channels_=out_channels_ , 
                            kernel_size_=1 , stride_=1 , padding_='valid' , 
                            bias_=True)
        )
        # TODO 当stride为1时padding the conv of TensorFlow equal to the conv of Pytorch
        # 
        #

        self.activate_ = nn.PReLU(out_channels_)
    
    def forward(self , x):
        input_ = x
        if self.stride_ > 1 :
            temp_ = self.max_pool_(input_)
        else :
            temp_ = input_
        
        if self.channelpad_ > 0 :
            temp_ = F.pad(temp_ , (0,0,0,0,0,self.channelpad_))
        
        result_ = self.activate_(self.convs_(input_) + temp_)

        return result_

class ResBlock(nn.Module):
    def __init__(self , in_channels) -> None:
        super(ResBlock , self).__init__()
        layers_ = [ResModule(in_channels , in_channels) for _ in range(4)]
        self.res_pipeline_ = nn.Sequential(*layers_)

    def forward(self , x):
        input_ = x
        result_ = self.res_pipeline_(input_)
        return result_

class PalmDetector(nn.Module):
    def __init__(self) -> None:
        super(PalmDetector , self).__init__()
        self.backbone_1 = nn.Sequential(
            ConvTensorFlow(in_channels_=3 , out_channels_=32 , kernel_size_=(5,5) , 
                        stride_=(2,2) , padding_="same") , 
            nn.PReLU(32) , 
            ResBlock(32) , 
            ResModule(in_channels_=32 , out_channels_=64 , stride_=2)
        )
    
    def forward(self , x):
        input_ = x
        temp_1 = self.backbone_1(input_)
        return temp_1
    

if __name__ == '__main__':
    device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')
    model_ = PalmDetector().to(device)
    input_dummy = torch.randn((1,3,192,192)).to(device)
    out = model_(input_dummy)
    print("out shape : " , out.shape)
    

