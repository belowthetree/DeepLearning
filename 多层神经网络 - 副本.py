import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as data
import torch
from torch.autograd import Variable

mnist = data.read_data_sets("MNIST_data/",one_hot = True)
in_units = 784
h1_units = 500
h2_units = 10

class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()
        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(3, 32, 3, 1, padding=1))
        layer1.add_module('relu1', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(32, 64, 3, 1, padding=1))
        layer2.add_module('relu2', nn.RelU(True))
        layer2.add_module('pool2', nn.MaxPool2d(2,2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(64, 128, 3, 1, padding=1))
        layer3.add_module('relu3', nn.RelU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2,2))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(2048, 512))
        layer4.add_module('fc_relu1', nn.ReLU(True))
        layer4.add_module('fc2', nn.Linear(512, 64))
        layer4.add_module('fc_relu2', nn.ReLU(True))
        layer4.add_module('fc3', nn.Linear(64, 10))
        self.layer4 = layer4

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        fc_input = conv3.view(conv3.size(0), -1)
        fc_out = self.layer4(fc_input)
        return fc_out

model = Conv_Net()

print(model)

#new_model = nn.Sequential(*list(model.children())[:2])

# for layer in model.named_modules():
#     if isinstance(layer[1], nn.Conv2d):
#         conv_model.add_module(layer[0], layer[1])

