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
    # input_data 为
    input_data = None
    # Read the model.
    with open('xxx.tflite', 'rb') as f:
        model_buffer = f.read()
    

    # 修改输出idx
    idx = 35  #可以通过interpreter.get_tensor_details()，查各层的idx值； 或者netron也可以看到
    model_buffer = buffer_change_output_tensor_to(model_buffer, idx)
    
    
    # 推理
    interpreter = tf.lite.Interpreter(model_content=model_buffer)
    interpreter.allocate_tensors()
    print("《--- interpreter.get_tensor_details() ---》")
    for i in range(len(interpreter.get_tensor_details())):
        print(f"{i} : " , interpreter.get_tensor_details()[i])
    
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    
    # 中间层的output值
    out_val = interpreter.get_tensor(output_index)
    print("《--- show out_val info ---》")
    print("out_val shape : " , out_val.shape)
    print("out_val type : " , type(out_val))
    print("np.unique(out_val) : " , np.unique(out_val))
    print("np.mean(out_val) : " , np.mean(out_val))
    

