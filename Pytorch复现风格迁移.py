import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from PIL import Image
from scipy.io import sio
import torch.nn.functional as F

STYLE_WEIGHT = 1
CONTENT_WEIGHT = 1
STYLE_LAYERS = ['relu1_2','relu2_2','relu3_2']
CONTENT_LAYERS = ['relu1_2']
_vgg_params = None
Path = ''

def vgg_params():
    global _vgg_params
    if _vgg_params is None:
        _vgg_params = sio.loadmat(Path)
    return _vgg_params

def vgg19(input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4','pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )
    weights = vgg_params()['layers'][0]
    net= input_image
    network = {}
    for i, name in enumerate(input_image):
        layer_type = name[:4]
        net = input_image
        if layer_type == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            kernels = np.transpose(kenels, (3,2,0,1))
            net = F.conv2d(net,kernels,stride=1,padding=(kernels.size()+1)//2)
        elif layer_type == 'pool':
            net = F.max_pool2d(net, kernel_size=2,stride=2,padding=1)
        network[name] = net
    return network

def content_loss(target_feature, content_feature):
    _,height,weight,channel = map(lambda i:i in content_feature.size())
    print ('content_features.get_shape() : ')
    print (content_features.get_shape())
    content_size = height*weight*channel
    return nn.L1Loss()(target_feature, content_feature)/content_size

def style_loss()