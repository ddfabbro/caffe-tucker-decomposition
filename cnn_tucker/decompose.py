import numpy as np
from sktensor import dtensor, tucker
from sktensor.core import ttm
import google.protobuf.text_format
from cnn_tucker.utils import caffe, decompose_layer, rename_nodes

def decompose_model(model_def_path, model_weights_path, layer_ranks):
    
    """ CREATE DECOMPOSED MODEL DEFINITION """
    
    with open(model_def_path) as f:
        model_def = caffe.proto.caffe_pb2.NetParameter()
        google.protobuf.text_format.Merge(f.read(), model_def)
        
    new_model_def = caffe.proto.caffe_pb2.NetParameter()
    new_model_def.name = model_def.name + '_decomposed'
    
    if model_def.input:
        new_model_def.input.extend(['data'])
        new_model_def.input_dim.extend(model_def.input_dim)
        
    new_layers = [] #Keeping track of new layers helps renaming nodes in the future
    
    for i, layer in enumerate(model_def.layer):
        if layer.name not in layer_ranks.keys() or layer.type != 'Convolution':
            new_model_def.layer.extend([layer])
        else:
            decomposed_layers = decompose_layer(layer, layer_ranks[layer.name])
            for i in range(3):
                new_layers.append(decomposed_layers[i].name)
            new_model_def.layer.extend(decomposed_layers)
            
    #Rename bottom/top nodes for some layers !!!
    new_model_def = rename_nodes(new_model_def, new_layers)
                
    new_model_def_path = model_def_path[:-9] + '_decomposed.prototxt'
    with open(new_model_def_path, 'w') as f:
        google.protobuf.text_format.PrintMessage(new_model_def, f)
        
    """ CREATE DECOMPOSED MODEL WEIGHTS """
    
    net = caffe.Net(model_def_path, model_weights_path, caffe.TEST)
    new_net = caffe.Net(new_model_def_path, model_weights_path, caffe.TEST)
    
    for conv_layer in layer_ranks.keys():
        weights = net.params[conv_layer][0].data
        bias = net.params[conv_layer][1].data
        T = dtensor(weights)
        rank = layer_ranks[conv_layer] + [T.shape[2], T.shape[3]]
        
        print('Decomposing %s...' %conv_layer)
        core, U = tucker.hooi(T, rank, init='nvecs')

        num_output = net.params[conv_layer][0].data.shape[0]
        channels = net.params[conv_layer][0].data.shape[1]
        
        core = ttm(core, U[3], mode=3)
        core = ttm(core, U[2], mode=2)
        Us = U[1].reshape(rank[1], channels, 1, 1)
        Ut = U[0].reshape(num_output, rank[0], 1, 1)
        
        np.copyto(new_net.params[conv_layer+'_S'][0].data, Us)
        np.copyto(new_net.params[conv_layer+'_core'][0].data, core)
        np.copyto(new_net.params[conv_layer+'_T'][0].data, Ut)
        np.copyto(new_net.params[conv_layer+'_T'][1].data, bias)
        
    new_model_weights_path = model_weights_path[:-11] + '_decomposed.caffemodel'
    new_net.save(new_model_weights_path)
    
    print('\nDecomposed model definition saved to %s' %new_model_def_path)
    print('\nDecomposed model weights saved to %s' %new_model_weights_path)
    print('\nPlease fine-tune')
    
    return new_model_def_path, new_model_weights_path
