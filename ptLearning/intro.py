import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.relu1 = nn.ReLU()
    self.max_pool1 = nn.MaxPool2d((2,2))

    self.conv2 = nn.Conv2d(6, 16, 5)
    self.relu2 = nn.ReLU()
    self.max_pool2 = nn.MaxPool2d((2,2))

    self.fc1 = nn.Linear(16*5*5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    #x = self.max_pool1(self.relu1(self.conv1(x)))
    x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
    x = self.max_pool2(self.relu2(self.conv2(x)))
    x = x.view(-1, self.num_flat_features(x))

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))

    return x

  def num_flat_features(self, x):
    dimension = x.size()[1:]
    mul = 1
    for i in dimension:
      mul *= i
    return mul
    

net = Net()

input = torch.randn(1, 1, 32, 32)
output = net.forward(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

net.zero_grad()

print('conv1.bias.grad before')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after')
print(net.conv1.bias.grad)

learning_rate = 0.01
for f in net.parameters():
  print(f.data.sub_(learning_rate*f.grad.data))
