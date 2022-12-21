import onnx
import onnx.helper as helper

model_path = r"models/palm_detection_full.onnx"
onnx_model = onnx.load(model_path)

graph = onnx_model.graph

print("# <------- graph -------> #")
print(graph.output)

print("# <------- node info -------> #")
print(graph.node[1])
print(graph.node[1].input[0])
print(graph.node[1].output[0])




print("# <------- create value_info -------> #")
prob_info = helper.make_tensor_value_info(
    name="output" , 
    elem_type=onnx.TensorProto.FLOAT,
    shape=[1,96,96,32]
)
graph.output.insert(0,prob_info)
print(" <------- insert prob_info -------> ")
print(graph.output)



import os
import onnx
import numpy as np
import onnxruntime as rt
from onnxsim import simplify
from onnx import shape_inference


def get_tensor_shape(tensor):
    dims = tensor.type.tensor_type.shape.dim
    n = len(dims)
    return [dims[i].dim_value for i in range(n)]

def runtime_infer(onnx_model):
    graph = onnx_model.graph
    input_shape = get_tensor_shape(graph.input[0])
    graph.output.insert(0, graph.input[0])
    for i, tensor in enumerate(graph.value_info):
        graph.output.insert(i + 1, tensor)
    model_file = "temp.onnx"
    onnx.save(onnx_model, model_file)

    sess = rt.InferenceSession(model_file)
    input_name = sess.get_inputs()[0].name
    input_data = np.ones(input_shape, dtype=np.float32)

    outputs = {}
    for out in sess.get_outputs():
        tensor = sess.run([out.name], {input_name: input_data})
        outputs[str(out.name)] = np.array(tensor[0]).shape
    os.remove(model_file)
    return outputs

def infer_shapes(model_file, running_mode=False):
    onnx_model = onnx.load(model_file)
    onnx.checker.check_model(onnx_model)
    inferred_onnx_model = shape_inference.infer_shapes(onnx_model)
        
    save_path = model_file[:-5] + "_new.onnx"
    onnx.save(inferred_onnx_model, save_path)
    print("Model is saved in:", save_path)

    outputs = {}
    if running_mode:
        outputs = runtime_infer(inferred_onnx_model)
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


model_1 = "./onnx/model_1.onnx"
outputs = infer_shapes(model_1, True)
print(outputs)

model_2 = "../onnx/models/model_2.onnx"
outputs = infer_shapes(model_2)
print(outputs)



