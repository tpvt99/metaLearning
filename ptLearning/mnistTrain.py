import sys
sys.path.insert(0, '/home/ai/metaL-Reproduction')

import numpy
from dataSet.mnist.mnistLoader import MnistDataset

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class CNN(nn.Module):
  def __init__(self, in_c, out_c, k_size, maxPoolSize):
    super(CNN, self).__init__()
    self.conv1d = nn.conv2d(in_channels = in_c, out_channels = out_c, kernel_size = k_size)
    self.relu1 = nn.ReLU()
    self.max_pool1 = nn.max_pool2d(maxPoolSize)

    self.conv2d = nn.conv2d(in_channels = out_c, out_channels = out_c, kernel_size = k_size)
    self.relu1 = nn.ReLU()
    self.max_pool1 = nn.max_pool2d(maxPoolSize)

    self.linear1 = nn.Linear()

  def forward(self, x):
    pass

# parameters

batchSize = 128
lr = 0.01
n = 1000

training_set = MnistDataset(isTrain = True, isTest = False)
training_loader = DataLoader(training_set, batch_size = batchSize, shuffle = True)

test_set = MnistDataset(isTrain = False, isTest = True)
test_loader = DataLoader(test_set, batch_size = batchSize, shuffle = False)
