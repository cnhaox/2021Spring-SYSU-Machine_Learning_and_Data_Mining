import numpy as np
import math
import time
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

BATCH_SIZE = 512
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, imageData, imageLabel):
        self.data = torch.from_numpy(imageData).float()
        self.data = self.data.permute(0, 3, 1, 2).contiguous()# (x,y,z,3)->(x,3,y,z)
        self.label = torch.from_numpy(imageLabel)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 10)
    def forward(self, x):
        inSize = x.size(0)
        out = x.view(inSize, -1)
        out = self.fc1(out)
        out = F.log_softmax(out, dim=1)
        return out


class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)
    def forward(self, x):
        inSize = x.size(0)
        out = x.view(inSize, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = F.log_softmax(out, dim=1)
        return out
    

class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        inSize = x.size(0)   
        out = F.relu(self.conv1(x))   # batch*3*32*32->batch*6*28*28
        out = F.max_pool2d(out, 2, 2) # batch*6*28*28->batch*6*14*14
        out = F.relu(self.conv2(out)) # batch*6*14*14->batch*16*10*10
        out = F.max_pool2d(out, 2, 2) # batch*16*10*10->batch*16*5*5
        out = out.view(inSize, -1)    # batch*16*5*5->batch*400
        out = F.relu(self.fc1(out))   # batch*400->batch*120
        out = F.relu(self.fc2(out))   # batch*120->batch*84
        out = self.fc3(out)           # batch*84->batch*10
        out = F.log_softmax(out, dim=1)
        return out


def train(model, device, train_loader, optimizer, epoch, lossList, accList):
    '''
    训练函数
    '''
    model.train()
    trainLoss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        with torch.no_grad():
            trainLoss += loss.item()
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%10 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    trainLoss /= math.ceil(len(train_loader.dataset)/BATCH_SIZE)
    lossList.append(trainLoss)
    accList.append(correct / len(train_loader.dataset))
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        lossList[-1], correct, len(train_loader.dataset),
        100. * accList[-1]))


def test(model, device, test_loader, lossList, accList, setType='Test'):
    '''
    测试函数
    '''
    model.eval()
    testLoss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            testLoss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    testLoss /= len(test_loader.dataset)
    lossList.append(testLoss)
    accList.append(correct / len(test_loader.dataset))
    print(setType+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        lossList[-1], correct, len(test_loader.dataset),
        100. * accList[-1]))


def fit(model, optimizer, trainLoader, validLoader, testLoader):
    trainLossList = list()
    trainAccList = list()
    validLossList = list()
    validAccList = list()
    testLossList = list()
    testAccList = list()
    startTime = time.time()
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, trainLoader, optimizer, epoch, trainLossList, trainAccList)
        test(model, DEVICE, validLoader, validLossList, validAccList, 'Valid')
        test(model, DEVICE, testLoader, testLossList, testAccList, 'Test')
    endTime = time.time()
    print('\n Total Time: {:.4f}\n'.format(endTime-startTime))


def main():
    # 读取数据集
    train_images = np.load("material/cifar10_train_images.npy")
    train_labels = np.load("material/cifar10_train_labels.npy")
    test_images = np.load("material/cifar10_test_images.npy")
    test_labels = np.load("material/cifar10_test_labels.npy")
    # [0,255]->[0,1]
    train_images = train_images / 255
    test_images = test_images / 255
    # 打乱训练集数据，划分验证集
    np.random.seed(0)
    state = np.random.get_state()
    np.random.shuffle(train_images)
    np.random.set_state(state)
    np.random.shuffle(train_labels)
    valid_images = train_images[int(train_images.shape[0]*0.8):,:,:,:]
    valid_labels = train_labels[int(train_labels.shape[0]*0.8):]
    train_images = train_images[:int(train_images.shape[0]*0.8),:,:,:]
    train_labels = train_labels[:int(train_labels.shape[0]*0.8)]
    # 显示训练集、验证集和测试集的shape
    print(train_images.shape) #(40000, 32, 32, 3)
    print(train_labels.shape) #(40000, )
    print(valid_images.shape)  #(10000, 32, 32, 3)
    print(valid_labels.shape)  #(10000, )
    print(test_images.shape)  #(10000, 32, 32, 3)
    print(test_labels.shape)  #(10000, )

    # 封装数据集
    trainDataset = MyDataset(train_images, train_labels)
    validDataset = MyDataset(valid_images, valid_labels)
    testDataset = MyDataset(test_images, test_labels)
    # 加载数据集
    trainLoader = DataLoader(trainDataset, batch_size = BATCH_SIZE, shuffle =True)
    validLoader = DataLoader(validDataset, batch_size = BATCH_SIZE, shuffle = True)
    testLoader = DataLoader(testDataset, batch_size = BATCH_SIZE, shuffle = True)

    # 加载模型和优化器
    linearModel = LinearNet().to(DEVICE)
    linearOptimizer = optim.Adam(linearModel.parameters())
    summary(linearModel, input_size=(3,32,32))
    # 训练与测试线性分类器
    fit(linearModel, linearOptimizer, trainLoader, validLoader, testLoader)

    # 加载模型和优化器
    mlpModel = MlpNet().to(DEVICE)
    mlpOptimizer = optim.Adam(mlpModel.parameters())
    summary(mlpModel, input_size=(3,32,32))
    # 训练与测试MLP
    fit(mlpModel, mlpOptimizer, trainLoader, validLoader, testLoader)

    # 加载模型和优化器
    cnnModel = CnnNet().to(DEVICE)
    cnnOptimizer = optim.Adam(cnnModel.parameters())
    summary(cnnModel, input_size=(3,32,32))
    # 训练与测试CNN
    fit(cnnModel, cnnOptimizer, trainLoader, validLoader, testLoader)
    

if __name__=='__main__':
    main()