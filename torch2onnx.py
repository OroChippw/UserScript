import torch
import warnings
import onnxruntime as ort

from collections import OrderedDict

from ConvTensorFlow import SingleConv
from SingleConv2d import SingleConv_torch

import numpy as np

seed = 123
torch.manual_seed(seed)      # 为CPU设置随机种子
torch.cuda.manual_seed(seed)    # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed) 

def load_weight(model , weight_path , device = "cpu"):
    checkpoint = torch.load(weight_path , map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else :
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers , discarded_layers = [] , []

    for k , v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else :
            discarded_layers.append(k)
    
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn('')
    else :
        print("successful")

    return model

def tar_pth2onnx(model , weight_path = None , 
                input_ = None, 
                onnx_save_path = r"result.onnx"):
    if weight_path is not None:
        model = load_weight(model , weight_path)
    
    model.eval()

    x = input_

    torch.onnx.export(
        model , 
        x ,
        onnx_save_path , 
        opset_version=11,
        input_names=["input_1"],
        do_constant_folding=True , 
        output_names=["output_1"]
    )


def main():
    model_tfsame = SingleConv()
    model_torch = SingleConv_torch()
    weight_path = None
    input_shape = torch.randn(1,3,192,192)
    
    #
    model_torch.backbone_1[0].weight.data = model_tfsame.backbone_1[0].weight.data
    model_torch.backbone_1[0].bias.data = model_tfsame.backbone_1[0].bias.data

    
    onnx_path_tflite = "SingConv.onnx"
    onnx_path_pytorch = "SingleConv_torch.onnx"

    tar_pth2onnx(model_tfsame , weight_path , input_shape , onnx_path_tflite)
    tar_pth2onnx(model_torch , weight_path , input_shape , onnx_path_pytorch)

    

    # model_tfsame run
    output1_1 = model_tfsame(input_shape)
    modelOnnx1_2 = ort.InferenceSession(onnx_path_tflite)
    onnx_input_name = modelOnnx1_2.get_inputs()[0].name
    output1_2 = modelOnnx1_2.run(None, {onnx_input_name:input_shape.detach().numpy()})


    # model_torch run
    output2_1 = model_torch(input_shape)
    modelOnnx2_2 = ort.InferenceSession(onnx_path_pytorch)
    onnx_input_name = modelOnnx2_2.get_inputs()[0].name
    output2_2 = modelOnnx2_2.run(None, {onnx_input_name:input_shape.detach().numpy()})

    print("------------- model_tfsame ------------")

    print("------> output1_1 mean:{}".format(np.mean(output1_1.detach().numpy())))
    print("------> output1_1 unique:{}".format(np.unique(output1_1.detach().numpy())))
    print("------> output1_2 mean:{}".format(np.mean(output1_2)))
    print("------> output1_2 unique:{}".format(np.unique(output1_2)))

    print("====="*10)
    print()

    print("------------- model_torch ------------")

    print("------> output2_1 mean:{}".format(np.mean(output2_1.detach().numpy())))
    print("------> output2_1 unique:{}".format(np.unique(output2_1.detach().numpy())))
    print("------> output2_2 mean:{}".format(np.mean(output2_2)))
    print("------> output2_2 unique:{}".format(np.unique(output2_2)))


# def debug():
#     input_dummy = torch.randn(1,3,192,192)

#     torch_output = SingleConv_torch(input_dummy)
#     torch_onnx_output = SingleConv_torch(input_dummy)

#     tflite_output = SingleConv(input_dummy)
#     tflite_onnx_output = SingleConv(input_dummy)


#     pass

if __name__ == '__main__':
    main()
    