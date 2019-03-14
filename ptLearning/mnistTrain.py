import sys
sys.path.insert(0, '/home/ai/metaL-Reproduction')

import numpy
from dataSet.mnist.mnistLoader import MnistDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class Meter():
  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n =1):
    self.val = val
    self.sum += self.val * n
    self.count += n
    self.avg = self.sum / self.count

class CNN(nn.Module):
  def __init__(self, in_c, out_c, k_size, maxPoolSize):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels = in_c, out_channels = 64, kernel_size = k_size)
    self.max_pool1 = nn.MaxPool2d(maxPoolSize)

    self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = k_size)
    self.max_pool2 = nn.MaxPool2d(maxPoolSize)

    self.drop_out = nn.Dropout(0.9)

    self.linear1 = nn.Linear(in_features = 5*5*128, out_features = 128)

    self.linear2 = nn.Linear(128, out_c)

  def forward(self, x):
    # conv1 layer
    x = self.max_pool1(F.relu(self.conv1(x)))

    # conv2 layer
    x = self.max_pool2(F.relu(self.conv2(x)))

    # dropout
    x = self.drop_out(x)

    # flatten
    x = x.view(-1, self.flatten(x))

    # linear 1
    x = F.relu(self.linear1(x))

    # linear 2
    x = self.linear2(x)

    return x

  def flatten(self, x):
    size = x.size()[1:]
    mul = 1
    for i in size:
      mul *= i
    return mul


def top1(output, target):
  with torch.no_grad():
    output = F.softmax(output, dim = 1)
    outputMax = torch.max(output, 1)[1]
    return sum(outputMax == target).item() / target.size(0)

# parameters

batchSize = 100
lr = 0.001
n = 10

training_set = MnistDataset(isTrain = True, isTest = False)
training_loader = DataLoader(training_set, batch_size = batchSize, shuffle = True)

test_set = MnistDataset(isTrain = False, isTest = True)
test_loader = DataLoader(test_set, batch_size = batchSize, shuffle = False)

CNNModel = CNN(1, 10, 3, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(CNNModel.parameters(), lr= lr, momentum = 0.9)

# Model parameters 

for epoch in range(n):
  losses = Meter()
  accuracy = Meter()

  CNNModel.train()

  for i, data in enumerate(training_loader):
    batchX, batchY = data

    optimizer.zero_grad()
    output = CNNModel(batchX)
    loss = criterion(output, batchY)

    loss.backward()
    optimizer.step()

    # update Meter
    losses.update(loss.item(), batchX.size(0))
    #print(top1(output, batchY))
    #print(batchX.size(0))
    accuracy.update(top1(output, batchY))

    print('Epoch: [{0}|{1}/{2}]\t'
          'Loss: {loss.val:.3f} ({loss.avg:.3f})\t'
          'Acc: {acc.val:2.2f} ({acc.avg:2.2f})\t'.format(
          epoch, batchSize*(i+1), len(training_loader)*batchSize,
          loss = losses, acc = accuracy))

  CNNModel.eval()

  losses = Meter()
  accuracy = Meter()

  with torch.no_grad():
    for i, data in enumerate(test_loader):
      batchX, batchY = data
      output = CNNModel(batchX)
        
      loss = criterion(output, batchY)

      losses.update(loss.item(), batchX.size(0))
      accuracy.update(top1(output, batchY))
      print('(Test) Epoch: [{0}|{1}/{2}]\t'
            'Loss: {loss.val:.3f} ({loss.avg:.3f})\t'
            'Acc: {acc.val:2.2f} ({acc.avg:2.2f})\t'.format(
            epoch, batchSize*(i+1), len(test_loader)*batchSize,
            loss = losses, acc = accuracy))



