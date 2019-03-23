import sys
sys.path.insert(0, '/home/ai/metaL-Reproduction')
import os
import random

import numpy as np
import matplotlib.image as mpimg
from figs import ABS_PATH

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

OMNIGLOT_DATAPATH = ABS_PATH + '/dataSet/omniglot/python'
IMAGES_PER_CHARACTER = 20


class OmniglotLoader(Dataset):
  def __init__(self, train, n):
    super(OmniglotLoader, self).__init__()
    self.data = None
    self.labels = None
    self.train = train

    self.get_batch(n)

  def load_images(self, target = "standard", path=OMNIGLOT_DATAPATH):
    """
    This functions load images from Omniglot
    Arguments:
        target: standard or minimal
        path: path to the Omniglot dataset
    """
    X = []
    Y = []
    folderName = {}
    if target == "standard":
        trainFolders = ["images_background"]
        testFolders = ["images_evaluation"]
    elif target == "minimal":
        trainFolders = ["images_background_small1", "images_background_small2"]
        testFolders = ["images_evaluation"]

    if self.train:
        for trainFolder in trainFolders:
            folderPath = os.path.join(path, trainFolder)

            imgAllCount = 0 # this is counted for the whole images in all alphabet
            chaAllCount = 0 # this is counted for the whole characters in all alphabet

            for alphabet in sorted(os.listdir(folderPath)):
                alphabetPath = os.path.join(folderPath, alphabet)
                folderName[alphabet] = {'totalChar': 0, 'charIndex': [], 'totalImg': 0, 'imgIndex': []}

                imgAlphabetCount = 0 # this is counted for the number of images in this alphabet
                chaAlphabetCount = 0 # this is counted for the number of character in this alphabet

                folderName[alphabet]['charIndex'].append(chaAllCount)
                folderName[alphabet]['imgIndex'].append(imgAllCount)

                for letter in sorted(os.listdir(alphabetPath)):
                    letterPath = os.path.join(alphabetPath, letter)

                    for letterImage in os.listdir(letterPath):
                        imagePath = os.path.join(letterPath, letterImage)
                        image = mpimg.imread(imagePath)
                        X.append(image)
                        Y.append(chaAllCount)

                        imgAlphabetCount += 1
                        imgAllCount += 1

                    chaAlphabetCount += 1
                    chaAllCount += 1

                folderName[alphabet]['totalChar'] = chaAlphabetCount
                folderName[alphabet]['totalImg']  = imgAlphabetCount
                folderName[alphabet]['charIndex'].append(chaAllCount-1)
                folderName[alphabet]['imgIndex'].append(imgAllCount-1)

        X = np.stack(X) 
        X = X.reshape(-1, IMAGES_PER_CHARACTER, X.shape[1], X.shape[2])
        return X, np.stack(Y), folderName

    def get_batch(self, n):
        X, Y, folderProp = self.load_images()
        numOfChars, _, w, h = X[0]
        # generate n groups in numOfCharacters
        groups = np.random.choice(numOfChars, n, replace = False)

        # generate pairs which is a list, n*2*105*105*1
        pairs = np.zeros((n,2) + X.shape[2:end] + (1,))
        
        # generate half n is same, half is not
        labels = np.random.choice(2, n, p = [0.5, 0.5])

        for i in range(n):
            index1 = np.random.randint(IMAGES_PER_CHARACTER)
            group = groups[i]
            pairs[i][0] = X[group, index1].reshape(w, h, 1)

            index2 = np.random.randin(IMAGES_PER_CHARACTER)

            if labels[i] == 1:
                pairs[i][1] = X[group, index2].reshape(w, h, 1)
            else:
                group = (group + np.random.randint(1,numOfChars) % numOfChars)
                pairs[i][1] = X[group, index2].rshape(w, h, 1)

        self.data = pairs
        self.labels = labels
   
   def __len__(self):
       return self.data.shape[1]

   def __get_item__(self, index):
       img1 = self.data[0][index]
       img2 = self.data[1][index]
       label = self.labels[index]

       return ([img1, img2], label)

    

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


#OmniglotLoader().load_images()
