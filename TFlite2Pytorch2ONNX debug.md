# TFlite2Pytorch2ONNX debug
[TOC]
## 1、模型搭建部分
### 1.1、TFlite中SAME/VALID卷积的Pytorch实现
TensorFlow在调用其卷积 `conv2d` 的时候，TensorFlow 有两种填充方式，分别是 **padding = 'SAME'** 和 **padding = 'VALID'**，其中前者是默认值。如果卷积的**步幅（stride）**取值为 **1**，那么 padding = 'SAME' 就是指特征映射的分辨率在卷积前后保持不变，而 padding = 'VALID' 则是要下降 **k - 1** 个像素（即不填充，k 是卷积核大小）。比如，对于长度为 **5** 的特征映射，如果卷积核大小为 **3**，那么两种填充方式对应的结果是

![img](https://upload-images.jianshu.io/upload_images/11381959-57d67bc9604a4d85.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

 

  在pytorch中，如果你不指定padding的大小，在pytorch中默认的padding方式就是vaild    

  当卷积的stride为1时，TFlite卷积输出结果和Pytorch的nn.Conv2d等效，但当卷积的步幅 **stride = 2**，则两者的结果会有差异，比如对于 **224x224** 分辨率的特征映射，指定 **k = 5**，虽然两者的结果都得到 **112x112** 分辨率的特征映射，但结果却是不同的。比如，在输入和权重都一样的情况下，我们得到结果（**运行后面给出的的代码：compare_conv.py，将第 22 行简化为：p = k // 2，将第 66/67 行注释掉**）![image-20221216151210206](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20221216151210206.png)

```python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:44:31 2019

@author: shirhe-lyh
"""

import numpy as np
import tensorflow as tf
import torch

tf.enable_eager_execution()

np.random.seed(123)
tf.random.set_seed(123)
torch.manual_seed(123)

h = 224
w = 224
k = 5
s = 2
p = k // 2 if s == 1 else 0 # line 22


x_np = np.random.random((1, h, w, 3))
x_tf = tf.constant(x_np)
x_pth = torch.from_numpy(x_np.transpose(0, 3, 1, 2))


def pad(x, kernel_size=3, dilation=1):
    """For stride = 2 or stride = 3"""
    pad_total = dilation * (kernel_size - 1) - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    x_padded = torch.nn.functional.pad(
        x, pad=(pad_beg, pad_end, pad_beg, pad_end))
    return x_padded


conv_tf = tf.layers.Conv2D(filters=16, 
                           padding='SAME',
                           kernel_size=k,
                           strides=(s, s))

# Tensorflow prediction
with tf.GradientTape(persistent=True) as t:
    t.watch(x_tf)
    y_tf = conv_tf(x_tf).numpy()
    print('Shape: ', y_tf.shape)
    
    
conv_pth = torch.nn.Conv2d(in_channels=3,
                           out_channels=16,
                           kernel_size=k,
                           stride=s,
                           padding=p)

# Reset parameters
weights_tf, biases_tf = conv_tf.get_weights()
conv_pth.weight.data = torch.tensor(weights_tf.transpose(3, 2, 0, 1))
conv_pth.bias.data = torch.tensor(biases_tf)


# Pytorch prediction
conv_pth.eval()
with torch.no_grad():
    if s > 1: # line 66 
        x_pth = pad(x_pth, kernel_size=k) # line 67
    y_pth = conv_pth(x_pth)
    y_pth = y_pth.numpy().transpose(0, 2, 3, 1)
    print('Shape: ', y_pth.shape)
    
    
# Compare results
print('y_tf: ')
print(y_tf[:, h//s-1, 0, :])
print('y_pth: ')
print(y_pth[:, h//s-1, 0, :])  
```

![image-20221216151411806](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20221216151411806.png)

  一个完整的TFlite卷积的封装实现方式如下代码所示，直接使用即可。经过测试，该实现方式与TFlite中的tflite.keras.layers.Conv2D实现的SAME/VALID在输出结果上，均值指标误差为[e^-6,e^-7]之间

```python
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
```
## 2、模型对齐
### 2.1、CUDNN、pytorch及numpy设置随机数种子
#### 2.1.1、CUDNN seed
  cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率，如果保证可重复性，可以使用如下配置，但实际上对于精度的影响不大，仅仅时小数点后几位的差别，不建议修改，会使计算效率降低
```python
from torch.backends import cudnn
cudnn.benchmark = False      # if benchmark=True, deterministic will be False
cudnn.deterministic = True
```
#### 2.1.2、Pytorch seed
```python
torch.manual_seed(seed)      # 为CPU设置随机种子
torch.cuda.manual_seed(seed)    # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
```
#### 2.1.3、Python & Numpy seed
```python
import random
import numpy as np
random.seed(seed)
np.random.seed(seed)
```
### 2.2、各框架抽取中间层输出结果
#### 2.2.1、Pytorch抽取中间层结果
Pytorch抽取中间层结果只需要在网络forward过程中抽取结果即可
#### 2.2.2、TFlite抽取中间层结果
TFlite查看当前网络模型所有节点的name、idx及对应的shape，可将输出结果重定向到文件里面方便查看
```python
with open('xxx.tflite', 'rb') as f:
        model_buffer = f.read()
interpreter = tf.lite.Interpreter(model_content=model_buffer)
    interpreter.allocate_tensors()
    for i in range(len(interpreter.get_tensor_details())):
        print(f"{i} : " , interpreter.get_tensor_details()[i])
```
输入结果如下图所示

![image-20221216143325793](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20221216143325793.png)

根据查阅上文输出结果得到目标中间层的idx后，调整output到指定idx下并进行推理的demo如下所示

```python
import flatbuffers
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb
 
def OutputsOffset(subgraph, j):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(subgraph._tab.Offset(8))
    if o != 0:
        a = subgraph._tab.Vector(o)
        return a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4)
    return 0
 
# 调整output到指定idx
def buffer_change_output_tensor_to(model_buffer, new_tensor_i):
    root = schema_fb.Model.GetRootAsModel(model_buffer, 0)
    output_tensor_index_offset = OutputsOffset(root.Subgraphs(0), 0)
    # Flatbuffer scalars are stored in little-endian.
    new_tensor_i_bytes = bytes([
    new_tensor_i & 0x000000FF, \
    (new_tensor_i & 0x0000FF00) >> 8, \
    (new_tensor_i & 0x00FF0000) >> 16, \
    (new_tensor_i & 0xFF000000) >> 24 \
    ])
    # Replace the 4 bytes corresponding to the first output tensor index
    return model_buffer[:output_tensor_index_offset] + new_tensor_i_bytes + model_buffer[output_tensor_index_offset + 4:]

def debug():
    # input_data 为前处理后需要输入进网络的值
    input_data = None
    # Read the model.
    with open('xxx.tflite', 'rb') as f:
        model_buffer = f.read()
   
    # 修改输出idx
    idx = 35
    model_buffer = buffer_change_output_tensor_to(model_buffer, idx)
    
    
    # 推理
    interpreter = tf.lite.Interpreter(model_content=model_buffer)
    interpreter.allocate_tensors()
    
    # 此处get_input_details()[0]代表第一个输入
    # get_output_details()[0]代表第一个输出
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    
    # 使用解释器查考对应输出index中间层的输出值即可
    out_val = interpreter.get_tensor(output_index)
    print("《--- show out_val info ---》")
    print("out_val shape : " , out_val.shape)
    print("out_val type : " , type(out_val))
    print("np.unique(out_val) : " , np.unique(out_val))
    print("np.mean(out_val) : " , np.mean(out_val))
```
#### 2.2.3、ONNX抽取中间层结果
ONNX抽取中间层结果需要在图结构中给每个算子添加一个输出头，具体实现代码如下
```python
import onnx
import numpy as np
import onnxruntime as ort
from onnx import shape_inference

def get_tensor_shape(tensor):
    dims = tensor.type.tensor_type.shape.dim
    n = len(dims)
    return [dims[i].dim_value for i in range(n)]

def runtime_infer(onnx_model ,save_file = "temp.onnx"):
    graph = onnx_model.graph
    input_shape = get_tensor_shape(graph.input[0])
    graph.output.insert(0, graph.input[0])
    for i, tensor in enumerate(graph.value_info):
        graph.output.insert(i + 1, tensor)
    model_file = save_file
    onnx.save(onnx_model, model_file)

def infer_shapes(model_file, running_mode=False , save_file = "temp.onnx"):
    onnx_model = onnx.load(model_file)
    onnx.checker.check_model(onnx_model)
    inferred_onnx_model = shape_inference.infer_shapes(onnx_model)
        
    outputs = {}
    if running_mode:
        outputs = runtime_infer(inferred_onnx_model , save_file=save_file)
    else:
        graph = inferred_onnx_model.graph
        # only 1 input tensor
        tensor = graph.input[0]
        outputs[str(tensor.name)] = get_tensor_shape(tensor)
        # process tensor
        for tensor in graph.value_info:
            outputs[str(tensor.name)] = get_tensor_shape(tensor)
        # output tensor
        for tensor in graph.output:
            outputs[str(tensor.name)] = get_tensor_shape(tensor)
    return outputs

def debug():
    model_path = r"models/palm_detection_full.onnx"
    save_file = r"models/temp.onnx"
    infer_shapes(model_path, True , save_file=save_file)
    model_ = ort.InferenceSession(save_file)
    input_data = ""
    pred_ = model_.run(None , {'input_name' : input_data})
    for i in range(len(pred_)):
        if i == 0: # 避免重复输出输入的信息
            continue
        print(f"pred{i} shape : " , pred_[i].shape)
        print(f"pred{i} unique : " , np.unique(pred_[i]))
        print(f"pred{i} mean : " , np.mean(pred_[i]))

if __name__ == '__main__':
    debug()
```
结果如图中黄色框所示，可以在每个算子间都新增了一个输出头，可以通过打印出来对应的形状及顺序查看中间层结果

![image-20221216143022842](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20221216143022842.png)

### 2.3、两框架抽取结果对齐指标（有待优化及补充）
前提：**在进行中间层抽取结果之前，请先对前处理输入结果进行对齐**，前处理过程如通道顺序未进行转换，其预处理结果求均值数值会对齐，然而后续输出的结果会错误
框架之间的算子转换会产生一定的误差，一般应用下对齐至e^-5到e^-7即可

#### 2.3.1、两结果分别求均值对比 
#### 2.3.2、两结果作差、作差求和
#### 2.3.3、转换为numpy使用unique查看输出结果最值范围