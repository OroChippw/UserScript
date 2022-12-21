import onnx
import onnx.helper as helper

model_path = r"models/palm_detection_full.onnx"
onnx_model = onnx.load(model_path)

# graph = onnx_model.graph

# print("# <------- graph -------> #")
# print(graph.output)

# print("# <------- node info -------> #")
# print(graph.node[1])
# print(graph.node[1].input[0])
# print(graph.node[1].output[0])


# print("# <------- create value_info -------> #")
# prob_info = helper.make_tensor_value_info(
#     name="output" , 
#     elem_type=onnx.TensorProto.FLOAT,
#     shape=[1,96,96,32]
# )
# graph.output.insert(0,prob_info)
# print(" <------- insert prob_info -------> ")
# print(graph.output)



import os
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
    


