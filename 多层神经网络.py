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
        layer1.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, stride=1,padding=1))
        layer1.add_module('norm1', nn.BatchNorm2d(32))
        layer1.add_module('relu1', nn.ReLU())
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1,padding=1))
        layer2.add_module('norm2', nn.BatchNorm2d(64))
        layer2.add_module('relu2', nn.ReLU())
        layer2.add_module('pool2', nn.MaxPool2d(2,2))
        self.layer2 = layer2

        self.drop = nn.Dropout(0.5)

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(3136, 1024))
        layer4.add_module('relu3', nn.ReLU())
        layer4.add_module('fc2', nn.Linear(1024, 10))
        self.layer4 = layer4

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        fc_input = conv2.view(conv2.size(0), -1)
        fc_out = self.layer4(fc_input)
        return fc_out

if torch.cuda.is_available():
    model = Conv_Net().cuda()
else:
    model = Conv_Net()
loss_ = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-4)

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

for i in range(10000):
    x_train, y_train = mnist.train.next_batch(200)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    if torch.cuda.is_available():
        x_train = Variable(x_train, volatile=True).cuda()
        y_train = Variable(y_train, volatile=True).long().cuda()
    else:
        x_train = Variable(x_train, volatile=True)
        y_train = Variable(y_train, volatile=True).long()
    optimizer.zero_grad()
    x_train = x_train.view(200, 1, 28, 28)
    out = model(x_train)
    print(out.size(), y_train.size())
    exit()
    loss = loss_(out, y_train)
    loss.backward()
    optimizer.step()

    if(i % 50 == 0):
        x_train, y_train = mnist.test.next_batch(200)
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        if torch.cuda.is_available():
            x_train = Variable(x_train, volatile=True).cuda()
            y_train = Variable(y_train, volatile=True).long().cuda()
        else:
            x_train = Variable(x_train, volatile=True)
            y_train = Variable(y_train, volatile=True).long()
        x_train = x_train.view(200, 1, 28, 28)
        out = model(x_train)
        loss = loss_(out, y_train)
        print('step :{} loss is {} acc is {:f}%'.format(i, loss, out.data.max(1)[1].eq(y_train.data).sum()/2))
