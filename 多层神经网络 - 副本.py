import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as data
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision

mnist = data.read_data_sets("MNIST_data/",one_hot = False)
in_units = 784
h1_units = 500
h2_units = 10

class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()
        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(1, 32, kernel_size=5, stride=1,padding=2))
        layer1.add_module('norm1', nn.BatchNorm2d(32))
        layer1.add_module('relu1', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(32, 64, kernel_size=5, stride=1,padding=2))
        layer2.add_module('norm2', nn.BatchNorm2d(64))
        layer2.add_module('relu2', nn.ReLU(True))
        layer2.add_module('pool2', nn.MaxPool2d(2,2))
        self.layer2 = layer2

        self.drop = nn.Dropout2d(0.5)

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(3136, 10))
        self.layer4 = layer4

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        fc_input = self.drop(conv2.view(conv2.size(0), -1))
        fc_out = self.layer4(fc_input)
        return fc_out


model = Conv_Net()
loss_ = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-5)

print(model)

#new_model = nn.Sequential(*list(model.children())[:2])

# for layer in model.named_modules():
#     if isinstance(layer[1], nn.Conv2d):
#         conv_model.add_module(layer[0], layer[1])

# for m in model.modules():
#     if isinstance(m, nn.Conv2d):
#         init.normal(m.weight.data)
#         init.xavier_normal(m.weight.data)
#         init.kaiming_normal(m.weight.data)
#         m.bias.data.fill_(0)
#     elif isinstance(m, nn.Linear):
#         m.weight.data.normal_()
    
for i in range(2000):
    x_train, y_train = mnist.train.next_batch(200)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_train = Variable(x_train, volatile=True)
    y_train = Variable(y_train, volatile=True).long()
    
    optimizer.zero_grad()
    x_train = x_train.view(200, 1, 28, 28)
    out = model(x_train)
    loss = loss_(out, y_train)
    loss.backward()
    optimizer.step()

    if(i % 50 == 0):
        print('step :{} loss is {} acc is {}%'.format(i, loss, out.data.max(1)[1].eq(y_train.data).sum()/2))


