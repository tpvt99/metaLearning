import sys
sys.path.insert(0, '/home/ai/metaL-Reproduction')
import os

import numpy as np
from figs import ABS_PATH

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

OMNIGLOT_DATAPATH = ABS_PATH + '/dataSet/omniglot/python'


class OmniglotLoader(Dataset):
  def __init__(self):
    super(OmniglotLoader, self).__init__()
    
  def load_images(self, target = "standard", path=OMNIGLOT_DATAPATH, train = True):
  """
  This functions load images from Omniglot
  Arguments:
    target: standard or minimal
    path: path to the Omniglot dataset
  """
    X = []
    Y = []

    for folder in os.listdir(path):
      folderPath = path + '/' + folder
      if target == "standard":
        trainFolder = ["images_background"]
        testFolder = ["images_evaluation"]
      elif target == "minimal":
        trainFolder= ["images_background_small1", "images_background_small2"]
        testFolder = ["images_evaluation"]
      if os.path.isdir(folderPath) and folder != '__MACOSX': 
        print("Loading images at folder: " + folderPath + "\n") 
        for alphabet in os.listdir(folderPath):
          alphabetPath = folderPath + '/' + alphabet
          print("Loading images at folder: " + alphabetPath)
  pass
    
    

class Network(nn.Module):
  def __init__(self, n_in,):
    super(Network, self).__init__()
    self.conv = nn.Sequential()
    self.conv.add_module("conv1", self.conv_create(1, 64, 10, 10))
    self.conv.add_module("conv2", self.conv_create(64, 128, 7, 7))
    self.conv.add_module("conv3", self.conv_create(128, 128, 4, 4))
    self.conv.add_module("conv4", self.conv_create(128, 256, 4, 4, False))

    self.fc1 = nn.Linear(6*6*256, 4096)
    self.sigmoid1 = nn.Sigmoid()

    self.fc2 = nn.Linear(4096, 1)
    self.sigmoid2 = nn.Sigmoid()

  def conv_create(self, n_in, n_out, kernelSize, maxPoolSize, isPool=True):
    layer = nn.Sequential()
    layer.add_module("conv2", nn.Conv2d(in_channels = n_in, out_channels = n_out, kernel_size = kernelSize))
    layer.conv2.weight = self._conv_w_init(0, 10e-2, layer.weight.size())
    layer.conv2.bias = self._conv_b_init(0.5, 10e-2, layer.bias.size())
    layer.add_module("relu", nn.ReLU())
    if isPool:
      layer.add_module("max_pool", nn.MaxPool2d(kernel_size = 2, stride = 2))

    return layer

  def fc_create(self, n_in, n_out):
    layer = nn.Sequential()
    layer.add("fc", nn.Linear())

  def _conv_w_init(mean, std, size):
    return torch.nn.Parameter(torch.from_numpy(np.random.normal(mean, std, size)))

  def _conv_b_init(mean, std, size):
    return torch.nn.Parameter(torch.from_numpy(np.random.normal(mean, std, size)))

  def flatten(self, x):
    size = x.size()[1:]

    dimension = 1
    for i in size:
      dimension *= i

    return dimension

  def forward_once(self, x):
    x = self.conv(x)
    x = x.view(-1, self.flatten(x))
    x = self.fc1(x)
    x = abs(net1 - net2)
    alpha = torch.nn.Parameter(torch.randn(x.size()))
    x = torch.mul(alpha, x)
    x = self.sigmoid1(x)
    x = self.fc2(x)
    x = self.sigmoid2(x)
    
  def forward(self, x, y):
    output1 = self.forward_once(x)
    output2 = self.forward_once(y)
    return [output1, output2]

class LossFunction(nn.Module):
  def __init__(self):
    super(LossFunction, self).__init__()


OmniglotLoader().load_images()
