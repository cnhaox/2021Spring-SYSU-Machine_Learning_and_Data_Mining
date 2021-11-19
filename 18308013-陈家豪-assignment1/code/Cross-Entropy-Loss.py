# coding: utf-8

def softmax(matrix, HandleOverflow = True):
    if HandleOverflow:
        ColMax = matrix.max(axis = 1)
        ColMax -= 1
        ColMax = np.reshape(ColMax, (ColMax.size, 1))
        temp = np.exp(matrix-ColMax)
    else:
        temp = np.exp(matrix)
    ColSum = temp.sum(axis=1)
    ColSum = np.reshape(ColSum, (ColSum.size,1))
    return temp/ColSum

def train(W, train_X, train_Y, Lambda, learning_rate):
    temp = softmax(np.dot(train_X, W)) - train_Y
    dW = (1/train_X.shape[0])*np.dot(train_X.T, temp) + 2*Lambda*W
    W -= learning_rate*dW
    return W

def predict(W, X):
    temp = softmax(np.dot(X, W))
    prediction = temp.argmax(axis=1)
    return prediction

def accuracy(prediction, labels_Y):
    current = 0
    for i in range(prediction.size):
        if labels_Y[i, prediction[i]] == 1:
            current += 1
    return current/prediction.size

def CELoss(W, X, Y):
    temp = softmax(np.dot(X, W))
    total = 0
    return -(1/Y.shape[0])*np.sum(np.log(temp)*Y)


def main():
    # 设置超参数
    LABEL_NUM = 10
    epochs = 100
    learning_rate = 0.005
    BATCH_SIZE = 512
    LAMBDA = 0.01
    K = 10

    # 获取训练集数据
    train_images = np.load("material/train-images.npy")
    train_labels = np.load("material/train-labels.npy")
    extend = np.ones(train_images.shape[0])
    train_X = np.c_[extend, train_images]
    train_Y = np.zeros((train_labels.size, LABEL_NUM))
    for i in range(train_labels.size):
        train_Y[i, train_labels[i]] = 1
    # 标准化
    meanVal = []
    stdVal = []
    for i in range(train_X.shape[1]):
        meanVal.append(np.mean(train_X[:, i]))
        stdVal.append(np.std(train_X[:,i]))
        if stdVal[i] != 0:
            train_X[:, i] = (train_X[:,i]-meanVal[i])/stdVal[i]

    # 获取测试集数据
    test_images = np.load("material/test-images.npy")
    test_labels = np.load("material/test-labels.npy")
    extend = np.ones(test_images.shape[0])
    test_X = np.c_[extend, test_images]
    test_Y = np.zeros((test_labels.size, LABEL_NUM))
    for i in range(test_labels.size):
        test_Y[i, test_labels[i]] = 1
    # 标准化
    for i in range(test_X.shape[1]):
        if stdVal[i] != 0:
            test_X[:, i] = (test_X[:,i]-meanVal[i])/stdVal[i]
    
    # 设置随机种子，初始化参数
    np.random.seed(0)
    W = np.random.rand(train_X.shape[1], LABEL_NUM)

    # 训练
    train_loss = []
    valid_loss = []
    train_acc = []
    test_acc =[]
    for i in range(epochs):
        # 打乱训练集数据和标签
        state = np.random.get_state()
        np.random.shuffle(train_X)
        np.random.set_state(state)
        np.random.shuffle(train_Y)
        # K折交叉验证
        block_size = int(train_X.shape[0]/K)
        train_loss.append(0)
        valid_loss.append(0)
        for k in range(K):
            # 获取训练集和验证集
            ValidBlock_X = train_X[k*block_size:(k+1)*block_size, :]
            ValidBlock_Y = train_Y[k*block_size:(k+1)*block_size, :]
            TrainBlock_X = np.r_[train_X[:k*block_size, :], train_X[(k+1)*block_size:, :]]
            TrainBlock_Y = np.r_[train_Y[:k*block_size, :], train_Y[(k+1)*block_size:, :]]
            # mini-batch
            for Idx in range(int(TrainBlock_X.shape[0]/BATCH_SIZE)):
                # 获取本次更新批次的索引
                StartIdx = Idx*BATCH_SIZE
                EndIdx = StartIdx+BATCH_SIZE
                if EndIdx > TrainBlock_X.shape[0]:
                    EndIdx = TrainBlock_X.shape[0]
                W = train(W, TrainBlock_X[StartIdx:EndIdx,:], TrainBlock_Y[StartIdx:EndIdx,:], LAMBDA, learning_rate)
            # 计算损失值
            train_loss[-1] += CELoss(W, TrainBlock_X, TrainBlock_Y)
            valid_loss[-1] += CELoss(W, ValidBlock_X, ValidBlock_Y)
        # 求平均值
        train_loss[-1] /= K
        valid_loss[-1] /= K
        if (i+1)%1 == 0:
            train_acc.append(accuracy(predict(W, train_X), train_Y))
            test_acc.append(accuracy(predict(W, test_X), test_Y))
            print("%d/%d:\tAvgTrainLoss = %.6f\tAvgValidLoss = %.6f\tTotalTrainAcc = %.6f\tTotalTestAcc = %.6f"%(i+1, epochs, train_loss[-1], valid_loss[-1], train_acc[-1], test_acc[-1]))
    
    # 画图
    plt.figure()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.xticks(range(0, epochs+1, 10))
    #plt.axes(yscale = "log")
    plt.plot(range(epochs), train_loss, c='red', label='train')
    plt.plot(range(epochs), valid_loss, c='green', label='valid')
    plt.legend()
    plt.grid()
    plt.savefig('CEloss.jpg', dpi = 1800)
    plt.show()

    plt.figure()
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.xticks(range(0, epochs+1, 10))
    plt.plot(range(epochs), train_acc, c='red', label='train')
    plt.plot(range(epochs), test_acc, c='blue', label='test')
    plt.legend()
    plt.grid()
    plt.savefig('CEacc.jpg', dpi = 1800)
    plt.show()

if __name__=='__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    main()
