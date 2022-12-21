import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 

class ConvTensorFlow(nn.Conv2d):
    def __init__(self, in_channels_ , out_channels_ , kernel_size_ , stride_ , 
                    padding_ = 'same', dilation_ = 1 , groups_ = 1 , bias_: bool = True) -> None:
        super(ConvTensorFlow , self).__init__(in_channels_, out_channels_, 
                    kernel_size_, stride_, 0 , dilation_ , groups_, bias_ )
        assert padding_.lower() in ('valid' , 'same') , \
            ValueError("padding must be 'same' or 'valid'")
        self.pad = padding_

    def compute_valid_shape(self , in_shape):
        # init template
        in_shape = np.asarray(in_shape).astype('int32')
        stride = np.asarray(self.stride).astype('int32')
        kernel_size = np.asarray(self.kernel_size).astype('int32')
        dilation = np.asarray(self.dilation).astype('int32')

        stride = np.concatenate([[1,1] , stride])
        kernel_size = np.concatenate([[1,1] , kernel_size])
        dilation = np.concatenate([[1,1] , dilation])

        if self.pad == 'same':
            out_shape = (in_shape + stride - 1) // stride
        else :
            out_shape = (in_shape - dilation * (kernel_size - 1) - 1) // stride + 1
        
        valid_input_shape = (out_shape - 1) * stride + 1 + dilation * (kernel_size - 1)

        return valid_input_shape
    
    def forward(self, input):
        in_shape = np.asarray(input.shape).astype('int32')
        valid_shape = self.compute_valid_shape(in_shape)
        pad = []
        for x in valid_shape - in_shape :
            if x == 0:
                continue
            pad_left = x // 2
            pad_right = x - pad_left
            # pad right should be larger tha pad left
            pad.extend((pad_left , pad_right))
        if np.not_equal(pad , 0).any():
            padded_input = F.pad(input , pad) 
        else :
            padded_input = input
        return super(ConvTensorFlow , self).forward(padded_input)

class SingleConv(nn.Module):
    def __init__(self) -> None:
        super(SingleConv , self).__init__()
        self.backbone_1 = nn.Sequential(
            ConvTensorFlow(in_channels_=3 , out_channels_=32 , kernel_size_=(5,5) , 
                        stride_=(1,1) , padding_="same") , 
        )
    
    def forward(self , x):
        input_ = x
        result_ = self.backbone_1(input_)
        return result_

def main():
    pass

if __name__ == '__main__':
    main()
    