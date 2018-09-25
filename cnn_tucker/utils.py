import caffe
import numpy as np
from VBMF import EVBMF

def conv_layer(name, num_output, kernel_size=1, pad=0, stride=1):
    layer = caffe.proto.caffe_pb2.LayerParameter()
    layer.type = 'Convolution'
    layer.name = name
    
    layer.convolution_param.num_output = num_output
    layer.convolution_param.kernel_size = kernel_size
    layer.convolution_param.pad = pad
    layer.convolution_param.stride = stride
    
    return layer

def decompose_layer(layer, rank):
    param = layer.convolution_param
    name = [layer.name+'_S', layer.name+'_core', layer.name+'_T']
    
    num_output = param.num_output
    kernel_size = param.kernel_size
    pad = param.pad if hasattr(param, 'pad') else 0
    stride = param.stride if hasattr(param, 'stride') else 1
    
    decomposed_layer = [
        conv_layer(name[0], rank[1]),
        conv_layer(name[1], rank[0], kernel_size, pad, stride),
        conv_layer(name[2], num_output),
    ]
    
    return decomposed_layer

def rename_nodes(model_def, new_layers):
    layer_index = len(model_def.layer)
    for i in range(layer_index):
        #Label Decomposed layers nodes
        if model_def.layer[i].name in new_layers:
            if i == 0:
                model_def.layer[i].bottom.extend(['data'])
            elif model_def.layer[i-1].type == 'ReLU':
                model_def.layer[i].bottom.extend([model_def.layer[i-2].name])
            elif model_def.layer[i-1].type in ['Convolution','Pooling']:
                model_def.layer[i].bottom.extend([model_def.layer[i-1].name])
            model_def.layer[i].top.extend([model_def.layer[i].name])
        #Rename Convolution layers nodes
        elif model_def.layer[i].type == 'Convolution':
            if model_def.layer[i-2].name in new_layers:
                model_def.layer[i].bottom[0] = model_def.layer[i-2].name
        #Rename ReLU layers nodes
        elif model_def.layer[i].type == 'ReLU':
            if model_def.layer[i-1].name in new_layers:
                model_def.layer[i].bottom[0] = model_def.layer[i-1].name
                model_def.layer[i].top[0] = model_def.layer[i-1].name
        #Rename Pooling layers nodes
        elif model_def.layer[i].type == 'Pooling':
            if model_def.layer[i-2].name in new_layers:
                model_def.layer[i].bottom[0] = model_def.layer[i-2].name
    
    return model_def

def estimate_ranks(weights):
    T0 = np.reshape(np.moveaxis(weights, 0, 0), (weights.shape[0], -1))
    T1 = np.reshape(np.moveaxis(weights, 1, 0), (weights.shape[1], -1))
    _, T0_rank, _, _ = EVBMF(T0)
    _, T1_rank, _, _ = EVBMF(T1)
    return [T0_rank.shape[0], T1_rank.shape[1]]
