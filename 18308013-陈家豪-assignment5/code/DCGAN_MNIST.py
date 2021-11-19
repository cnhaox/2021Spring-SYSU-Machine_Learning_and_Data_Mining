import numpy as np
import math
import time
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import ConvTranspose2d
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms,utils
from torchsummary import summary
import matplotlib.pyplot as plt
import imageio

BATCH_SIZE = 256
EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INT_DIMENSION = 1000
TRANS_CONV_SIZE = 16
CONV_SIZE = 16
CHANNEL_NUM = 1

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.convTranspose1 = nn.ConvTranspose2d(INT_DIMENSION, TRANS_CONV_SIZE*4, 3, 1, 0, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(TRANS_CONV_SIZE*4)
        self.convTranspose2 = nn.ConvTranspose2d(TRANS_CONV_SIZE*4, TRANS_CONV_SIZE*2, 3, 2, 0, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(TRANS_CONV_SIZE*2)
        self.convTranspose3 = nn.ConvTranspose2d(TRANS_CONV_SIZE*2, TRANS_CONV_SIZE, 4, 2, 1, bias=False)
        self.batchNorm3 = nn.BatchNorm2d(TRANS_CONV_SIZE)
        self.convTranspose4 = nn.ConvTranspose2d(TRANS_CONV_SIZE, CHANNEL_NUM, 4, 2, 1, bias=False)

    def forward(self, x):
        out = self.convTranspose1(x) # (batch, INT_DIMENSION, 1, 1, 1)->(batch, TRANS_CONV_SIZE*4, 3, 3)
        out = self.batchNorm1(out)
        out = F.relu(out)
        out = self.convTranspose2(out) # (batch, TRANS_CONV_SIZE*4, 3, 3)->(batch, TRANS_CONV_SIZE*2, 7, 7)
        out = self.batchNorm2(out)
        out = F.relu(out)
        out = self.convTranspose3(out) # (batch, TRANS_CONV_SIZE*2, 7, 7)->(batch, TRANS_CONV_SIZE, 14, 14)
        out = self.batchNorm3(out)
        out = F.relu(out)
        out = self.convTranspose4(out) # (batch, TRANS_CONV_SIZE, 14, 14)->(batch, CHANNEL_NUM, 28, 28)
        out = torch.tanh(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(CHANNEL_NUM, CONV_SIZE, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(TRANS_CONV_SIZE, CONV_SIZE*2, 4, 2, 1, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(CONV_SIZE*2)
        self.conv3 = nn.Conv2d(TRANS_CONV_SIZE*2, CONV_SIZE*4, 3, 2, 0, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(CONV_SIZE*4)
        self.conv4 = nn.Conv2d(CONV_SIZE*4, 1, 3, 1, 0, bias=False)

    def forward(self, x):
        out = self.conv1(x) # (batch, channels, 28, 28)->(batch, TRANS_CONV_SIZE, 14, 14)
        out = F.leaky_relu(out, 0.2, True)
        out = self.conv2(out) # (batch, TRANS_CONV_SIZE, 14, 14)->(batch, TRANS_CONV_SIZE*2, 7, 7)
        out = self.batchNorm1(out)
        out = F.leaky_relu(out, 0.2, True)
        out = self.conv3(out) # (batch, TRANS_CONV_SIZE*2, 7, 7)->(batch, TRANS_CONV_SIZE*4, 3, 3)
        out = self.batchNorm2(out)
        out = F.leaky_relu(out, 0.2, True)
        out = self.conv4(out) # (batch, TRANS_CONV_SIZE*4, 3, 3)->(batch, 1, 1, 1)
        out = torch.sigmoid(out)
        return out

def initWeight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train(generator_model, discriminator_model, device, train_loader, generator_optimizer, discriminator_optimizer, epoch, gLossList, dLossList, realScoreList, fakeScoreList1, fakeScoreList2):
    '''
    训练函数
    '''
    generator_model.train()
    discriminator_model.train()
    gLossSum = 0
    dLossSum = 0
    realScoreSum = 0
    fakeScore1Sum = 0
    fakeScore2Sum = 0
    iters = 0
    for batch_idx, (data,_) in enumerate(train_loader):
        # 更新判别器参数
        # discriminator_model.zero_grad()
        discriminator_optimizer.zero_grad()
        # 计算真实图片的loss
        data = data.to(device)
        # 生成数据标签,真实label=1
        dataNum = data.size(0)
        realLabel = torch.ones(dataNum).to(device)
        # 将真实数据输入判别器
        realOutput = discriminator_model(data).view(-1)
        realLoss = F.binary_cross_entropy(realOutput, realLabel)
        realScore = realOutput.mean().item()
        # 计算伪造图片的loss
        # 生成随机噪声
        noise = torch.randn(dataNum, INT_DIMENSION,1,1).to(device)
        # 获得虚假图片
        fakeData = generator_model(noise)
        # 生成数据标签，虚假label=0
        fakeLabel = torch.zeros(dataNum).to(device)
        # 将虚假数据输入判别器
        fakeOutput = discriminator_model(fakeData.detach()).view(-1)
        fakeLoss = F.binary_cross_entropy(fakeOutput, fakeLabel)
        fakeScore1 = fakeOutput.mean().item()
        # 计算loss，更新梯度
        dLoss = realLoss + fakeLoss
        dLoss.backward()
        discriminator_optimizer.step()

        # 更新生成器参数
        # generator_model.zero_grad()
        generator_optimizer.zero_grad()
        # 获取生成图片在判别器的输出值
        output = discriminator_model(fakeData).view(-1)
        # 计算loss，更新梯度
        gLoss = F.binary_cross_entropy(output, realLabel)
        gLoss.backward()
        fakeScore2 = output.mean().item()
        generator_optimizer.step()

        # 输出结果
        if (batch_idx+1)%50==0:
            print('[{}/{}][{}/{}]\tLoss_D: {:.4f}\tLoss_G: {:.4f}\tD(x): {:.4f}\tD(G(z)): {:.4f}/{:.4f}'.format(
                    epoch, EPOCHS, batch_idx+1, len(train_loader),
                    dLoss.item(), gLoss.item(), realScore, fakeScore1, fakeScore2))
        gLossSum += gLoss.item()
        dLossSum += dLoss.item()
        realScoreSum += realScore
        fakeScore1Sum += fakeScore1
        fakeScore2Sum += fakeScore2
        iters += 1
    gLossList.append(gLossSum/iters)
    dLossList.append(dLossSum/iters)
    realScoreList.append(realScoreSum/iters)
    fakeScoreList1.append(fakeScore1Sum/iters)
    fakeScoreList2.append(fakeScore2Sum/iters)

def test(generator_model, discriminator_model, device, test_loader, gLossList, dLossList, realScoreList, fakeScoreList):
    '''
    训练函数
    '''
    generator_model.train()
    discriminator_model.train()
    gLossSum = 0
    dLossSum = 0
    fakeScoreSum = 0
    realScoreSum = 0
    iters = 0
    with torch.no_grad():
        for (data,_) in test_loader:
            # 计算真实图片的loss
            data = data.to(device)
            # 生成数据标签,真实label=1
            dataNum = data.size(0)
            realLabel = torch.ones(dataNum).to(device)
            # 将真实数据输入判别器
            realOutput = discriminator_model(data).view(-1)
            realLoss = F.binary_cross_entropy(realOutput, realLabel)
            realScore = realOutput.mean().item()
            # 计算伪造图片的loss
            # 生成随机噪声
            noise = torch.randn(dataNum, INT_DIMENSION,1,1).to(device)
            # 获得虚假图片
            fakeData = generator_model(noise)
            # 生成数据标签，虚假label=0
            fakeLabel = torch.zeros(dataNum).to(device)
            # 将虚假数据输入判别器
            fakeOutput = discriminator_model(fakeData).view(-1)
            fakeLoss = F.binary_cross_entropy(fakeOutput, fakeLabel)
            fakeScore = fakeOutput.mean().item()
            # 计算loss，更新梯度
            dLoss = realLoss + fakeLoss
            gLoss = F.binary_cross_entropy(fakeOutput, realLabel)
            # 输出结果
            gLossSum += gLoss.item()
            dLossSum += dLoss.item()
            fakeScoreSum += fakeScore
            realScoreSum += realScore
            iters += 1
        gLossList.append(gLossSum/iters)
        dLossList.append(dLossSum/iters)
        fakeScoreList.append(fakeScoreSum/iters)
        realScoreList.append(realScoreSum/iters)
        print('test\tLoss_D: {:.4f}\tLoss_G: {:.4f}\tD(x): {:.4f}\tD(G(z)): {:.4f}'.format(
                    dLossList[-1], gLossList[-1], realScoreList[-1], fakeScoreList[-1]))

def generate(generator_model, noise):
    with torch.no_grad():
        fakeImg = generator_model(noise).cpu()
    return utils.make_grid(fakeImg, padding=2, normalize=True)

def fit(generator_model, discriminator_model, generator_optimizer, discriminator_optimizer, trainLoader, testLoader, fixedNoise):
    train_gLossList = list()
    train_dLossList = list()
    train_realScoreList = list()
    train_fakeScoreList1 = list()
    train_fakeScoreList2 = list()
    test_gLossList = list()
    test_dLossList = list()
    test_realScoreList = list()
    test_fakeScoreList = list()
    imgList = list()
    for epoch in range(1, EPOCHS + 1):
        train(generator_model, discriminator_model, DEVICE, 
              trainLoader, generator_optimizer, discriminator_optimizer, 
              epoch, train_gLossList, train_dLossList, train_realScoreList,
              train_fakeScoreList1, train_fakeScoreList2)
        test(generator_model, discriminator_model, DEVICE, 
              testLoader, test_gLossList, test_dLossList, test_realScoreList,
              test_fakeScoreList)
        imgList.append(generate(generator_model, fixedNoise))
    return (train_gLossList,train_dLossList,train_realScoreList,
     train_fakeScoreList1,train_fakeScoreList2,
     test_gLossList,test_dLossList,
     test_realScoreList,test_fakeScoreList,imgList
    )

def main():
    # 加载数据
    dataTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    trainDatasets = datasets.MNIST(root='./data', train=True, download=True,
                                     transform=dataTransform)
    testDatasets = datasets.MNIST(root='./data', train=False, download=True,
                                     transform=dataTransform)                              
    trainLoader = DataLoader(trainDatasets, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = DataLoader(testDatasets, batch_size=BATCH_SIZE, shuffle=True)
    # 查看真实图像
    plt.figure()
    realData = next(iter(trainLoader))
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(utils.make_grid(realData[0].to(DEVICE)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig("output/real_images.png", dpi=400)
    plt.show()
    # 设置随机种子
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # 加载模型和优化器
    generator = Generator().to(DEVICE)
    generator.apply(initWeight)
    summary(generator, input_size=(INT_DIMENSION,1,1))
    generatorOptimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5,0.999))

    discriminator = Discriminator().to(DEVICE)
    discriminator.apply(initWeight)
    summary(discriminator, input_size=(CHANNEL_NUM,28,28))
    discriminatorOptimizer = optim.Adam(discriminator.parameters(),lr=0.0002, betas=(0.5,0.999))
    # 训练拟合
    fixedNoise = torch.randn(64, INT_DIMENSION, 1, 1, device=DEVICE)# 固定噪声
    (train_gLossList,train_dLossList,train_realScoreList,
     train_fakeScoreList1,train_fakeScoreList2,
     test_gLossList,test_dLossList,
     test_realScoreList,test_fakeScoreList,imgList
    ) = fit(generator, discriminator, generatorOptimizer, discriminatorOptimizer, trainLoader, testLoader, fixedNoise)
    # 生成曲线图
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(range(EPOCHS), train_gLossList, label="generator train")
    plt.plot(range(EPOCHS), train_dLossList, label="discriminator train")
    plt.plot(range(EPOCHS), test_gLossList, label="generator test")
    plt.plot(range(EPOCHS), test_dLossList, label="discriminator test")
    plt.legend()
    plt.grid()
    plt.savefig("output/loss.png", dpi=400)
    plt.show()
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(range(EPOCHS), train_realScoreList, label="D(x) train")
    plt.plot(range(EPOCHS), train_fakeScoreList1, label="D1(G(z)) train")
    plt.plot(range(EPOCHS), train_fakeScoreList2, label="D2(G(z)) train")
    plt.plot(range(EPOCHS), test_realScoreList, label="D(x) test")
    plt.plot(range(EPOCHS), test_fakeScoreList, label="D(G(z)) test")
    plt.legend()
    plt.grid()
    plt.savefig("output/score.png", dpi=400)
    plt.show()
    # 比较真实图片和生成图片
    plt.figure()
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(utils.make_grid(realData[0].to(DEVICE)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(generate(generator, fixedNoise),(1,2,0)))
    plt.savefig("output/compare.png", dpi=400)
    plt.show()
    #将图片写入writer，生成gif
    with imageio.get_writer("output/output.gif", mode='I') as writer:
        for i in range(0,len(imgList)):
            writer.append_data(np.transpose(imgList[i],(1,2,0)).numpy())

if __name__=='__main__':
    main()