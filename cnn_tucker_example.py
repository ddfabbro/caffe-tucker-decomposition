import os
import urllib
from collections import OrderedDict
import cnn_tucker as tucker

ROOT_DIR = '/path/to/caffe-tucker-decomposition/'
os.chdir(ROOT_DIR)

if not os.path.isfile('models/VGG_ILSVRC_16_layers_deploy.caffemodel'):
    caffemodel_url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
    urllib.urlretrieve (caffemodel_url, 'models/VGG_ILSVRC_16_layers_deploy.caffemodel')

model = {
    'def': 'models/VGG_ILSVRC_16_layers_deploy.prototxt',
    'weights': 'models/VGG_ILSVRC_16_layers_deploy.caffemodel',
}

net_dict = tucker.utils.net_to_dict(model['def'], model['weights'])

#You can choose ranks arbitrarily
ranks4_1 = [200, 100]
#...based on some heuristic (e.g T/3 and S/3, where S == T == 256)
ranks4_2 = [170, 170]
#...or estimate via VBMF
ranks4_3 = tucker.utils.estimate_ranks(net_dict['conv4_3']['weights'])

layer_ranks = OrderedDict([
    ('conv4_1', ranks4_1),
    ('conv4_2', ranks4_2),
    ('conv4_3', ranks4_3),
])

data, _ = tucker.decompose_model(model['def'], model['weights'], layer_ranks)
