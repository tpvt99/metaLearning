import numpy as np
import os
from mnist import MNIST

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from figs import ABS_PATH

IMAGE_SIZE = 28 # 28 x 28
IMAGE_CHANNEL = 1

class MnistDataset(Dataset):
  def __init__(self, isTrain, isTest):
    """
    Initialize the data and labels based on the input of user
    only isTrain or isTest is True
    """
    x = MNIST(ABS_PATH + '/dataSet/mnist/')
    if isTrain:
      data = x.load_training()
    elif isTest:
      data = x.load_testing()
    self.data = data[0]
    self.label = data[1]

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.data)

  def __getitem__(self, index):

    x = torch.tensor(self.data[index], dtype = torch.float)
    x = x.view(IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
    y = torch.tensor(self.label[index])

    return x,y
