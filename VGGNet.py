import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as tf

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
train_dict = unpickle(r'.\cifar-10-batches-py\data_batch_1')
train_dict.update(unpickle(r'.\cifar-10-batches-py\data_batch_2'))
test_dict = unpickle(r'.\cifar-10-batches-py\test_batch')

class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #16*16*64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #8*8*128
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #4*4*256
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #2*2*512
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #1024
        )
        self.classifier = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 10),
        #nn.ReLU(True),
        #nn.Dropout(),
        #nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
#超参数
num_classes = 10
num_epochs = 20000
learning_rate = 1e-4
batch_size = 50

if torch.cuda.is_available():
    model = VGG(num_classes).cuda()
else:
    model = VGG(num_classes)
#预处理  
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight.data)
loss_ = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

x_train = torch.Tensor(train_dict[b'data'])
x_train = x_train.view(x_train.size(0), -1, 3)
y_train = torch.Tensor(train_dict[b'labels'])
y_train = y_train.view(y_train.size(0), -1)
x_test = torch.Tensor(test_dict[b'data'])
x_test = x_test.view(x_test.size(0), -1, 3)
y_test = torch.Tensor(test_dict[b'labels'])
y_test = y_test.view(y_test.size(0), -1)

for epoch in range(num_epochs - batch_size):
    x = x_train[epoch*batch_size:(epoch*batch_size+batch_size)]
    y = y_train[epoch*batch_size:(epoch*batch_size+batch_size)]
    if torch.cuda.is_available():
        x = Variable(x, volatile=True).cuda()
        y = Variable(y, volatile=True).long().cuda()
    else:
        x = Variable(x, volatile=True)
        y = Variable(y, volatile=True).long()
    x = x.view(x.size(0),3,32,32)
    y = y.view(y.size(0))
    out = model(x)
    loss = loss_(out, y)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        x = x_test[epoch*batch_size:(epoch*batch_size+batch_size)]
        y = y_test[epoch*batch_size:(epoch*batch_size+batch_size)]
        if torch.cuda.is_available():
            x = Variable(x, volatile=True).cuda()
            y = Variable(y, volatile=True).long().cuda()
        else:
            x = Variable(x, volatile=True)
            y = Variable(y, volatile=True).long()
        x = x.view(x.size(0),3,32,32)
        y = y.view(y.size(0))
        out = model(x)
        loss = loss_(out, y)
        print('step {}, loss is {}, acc is {}%'.format(epoch, loss, (out.data.max(1)[1].eq(y.data)).sum()*2))