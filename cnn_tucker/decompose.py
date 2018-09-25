import numpy as np
from sktensor import dtensor
from sktensor.tucker import hooi
from sktensor.core import ttm
import google.protobuf.text_format
from utils import caffe, decompose_layer, rename_nodes, net_to_dict

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
    
    for layer in model_def.layer:
        if layer.name not in layer_ranks.keys() or layer.type != 'Convolution':
            new_model_def.layer.extend([layer])
        else:
            decomposed_layer = decompose_layer(layer, layer_ranks[layer.name])
            for i in range(3):
                new_layers.append(decomposed_layer[i].name)
            new_model_def.layer.extend(decomposed_layer)
            
    #Rename bottom/top nodes for some layers !!!
    new_model_def = rename_nodes(new_model_def, new_layers)
                
    new_model_def_path = model_def_path[:-9] + '_decomposed.prototxt'
    with open(new_model_def_path, 'w') as f:
        google.protobuf.text_format.PrintMessage(new_model_def, f)
        
    """ CREATE DECOMPOSED MODEL WEIGHTS """
    
    #Convert caffemodel to dictionary
    net_dict = net_to_dict(model_def_path, model_weights_path)
    decomposed_net = caffe.Net(new_model_def_path, model_weights_path, caffe.TEST)
    data_dict = {} #Data containing fit, n_itr and exectimes for each layer
    
    for conv_layer in layer_ranks.keys():
        data_dict[conv_layer] = {}
        
        T = dtensor(net_dict[conv_layer]['weights'])
        rank = layer_ranks[conv_layer]
        kernel = [T.shape[2], T.shape[2]]
        
        print('\nDecomposing %s...' %conv_layer)
        core, U, fit, n_itr, exectimes = hooi(T, rank + kernel, init='nvecs')
        print('Reconstruction: %f%%' %(fit*100))
        print('Elapsed time: %.2fs' %sum(exectimes))
        
        data_dict[conv_layer]['fit'] = fit
        data_dict[conv_layer]['n_itr'] = n_itr
        data_dict[conv_layer]['exectime'] = sum(exectimes)
        
        num_output = net_dict[conv_layer]['weights'].shape[0]
        channels = net_dict[conv_layer]['weights'].shape[1]
        bias = net_dict[conv_layer]['bias']
        
        core = ttm(core, U[3], mode=3)
        core = ttm(core, U[2], mode=2)
        Us = U[1].reshape(rank[1], channels, 1, 1)
        Ut = U[0].reshape(num_output, rank[0], 1, 1)
        
        np.copyto(decomposed_net.params[conv_layer+'_S'][0].data, Us)
        np.copyto(decomposed_net.params[conv_layer+'_core'][0].data, core)
        np.copyto(decomposed_net.params[conv_layer+'_T'][0].data, Ut)
        np.copyto(decomposed_net.params[conv_layer+'_T'][1].data, bias)
        
    new_model_weights_path = model_weights_path[:-11] + '_decomposed.caffemodel'
    decomposed_net.save(new_model_weights_path)
    
    return data_dict, [new_model_def_path, new_model_weights_path]
