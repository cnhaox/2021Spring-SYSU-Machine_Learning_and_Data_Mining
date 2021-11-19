# coding: utf-8
def sigmoid(x):
    if x > 0:
        return 1/(1+np.exp(-x))
    else:
        return np.exp(x)/(np.exp(x)+1)

def train(func, W, train_X, train_Y, Lambda, learning_rate):
    if func=='hinge':
        temp = np.dot(train_X, W)*train_Y
        temp2 = np.zeros_like(temp)
        for i in range(temp.shape[0]):
            if temp[i]<1:
                temp2[i,0] = -1
        dW = (1/train_X.shape[0])*np.dot(train_X.T, temp2*train_Y) + Lambda*W
        W -= learning_rate*dW
        return W
    elif func=='cross':
        temp = np.dot(train_X, W)
        for i in range(temp.shape[0]):
            temp[i,0] = sigmoid(temp[i,0])
        dW = (1/train_X.shape[0])*np.dot(train_X.T, temp-train_Y) + Lambda*W
        W -= learning_rate*dW
        return W

def predict(func, W, X):
    prediction = np.dot(X, W)
    if func=='hinge':
        for i in range(prediction.shape[0]):
            if prediction[i,0]>0:
                prediction[i,0] = 1
            elif prediction[i,0]<0:
                prediction[i,0] = -1
        return prediction
    elif func=='cross':
        for i in range(prediction.shape[0]):
            if prediction[i,0]>=0:
                prediction[i,0] = 1
            else:
                prediction[i,0] = 0
        return prediction

def accuracy(prediction, labels_Y):
    current = 0
    for i in range(prediction.size):
        if int(prediction[i,0])==int(labels_Y[i,0]):
            current += 1
    return current/prediction.size

def HingeLoss(W, X, Y):
    temp = np.dot(X, W)*Y
    loss = 0.0
    for i in range(temp.shape[0]):
        if temp[i,0]<1:
            loss += 1-temp[i,0]
    return loss/X.shape[0]

def CrossEntropyLoss(W, X, Y):
    temp = np.dot(X, W)
    loss = 0.0
    for i in range(temp.shape[0]):
        if int(Y[i,0]) == 1:
            loss += np.log(sigmoid(temp[i,0]))
        else:
            loss += np.log(1-sigmoid(temp[i,0]))
    loss *= -(1/X.shape[0])
    return loss

def main():
    # 超参数设置
    epochs = 100
    learning_rate = 0.005
    BATCH_SIZE = 512
    LAMBDA = 0.01
    K = 10

    # 获取训练集数据
    train_images = np.load("material/train-images.npy")
    train_labels = np.load("material/train-labels.npy")
    # 添加偏置1
    extend = np.ones(train_images.shape[0])
    train_X = np.c_[extend, train_images]
    # {0, 1}标签类型
    train_Y = np.reshape(train_labels, (-1,1))
    # {-1, 1}标签类型
    train_Y2 = np.zeros((train_labels.size,1))
    for i in range(train_labels.size):
        if train_labels[i]==1:
            train_Y2[i,0] = 1
        else:
            train_Y2[i,0] = -1
    # 特征标准化
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
    # 添加偏置1
    extend = np.ones(test_images.shape[0])
    test_X = np.c_[extend, test_images]
    # {0, 1}标签类型
    test_Y = np.reshape(test_labels, (-1,1))
    # {-1, 1}标签类型
    test_Y2 = np.zeros((test_labels.size,1))
    for i in range(test_labels.size):
        if test_labels[i]==1:
            test_Y2[i,0] = 1
        else:
            test_Y2[i,0] = -1
    # 特征标准化
    for i in range(test_X.shape[1]):
        if stdVal[i] != 0:
            test_X[:, i] = (test_X[:,i]-meanVal[i])/stdVal[i]
    
    # 设置参数矩阵
    np.random.seed(0)
    W = np.random.rand(train_X.shape[1], 1)
    # cross-entropy loss模型训练
    train_loss = []
    valid_loss = []
    train_acc = []
    test_acc =[]
    time_start = time.time()
    for i in range(epochs):
        # 打乱训练集数据和标签
        state = np.random.get_state()
        np.random.shuffle(train_X)
        np.random.set_state(state)
        np.random.shuffle(train_Y)
        np.random.set_state(state)
        np.random.shuffle(train_Y2)
        # K折交叉验证
        block_size = int(train_X.shape[0]/K)
        train_loss.append(0)
        valid_loss.append(0)
        for k in range(K):
            ValidBlock_X = train_X[k*block_size:(k+1)*block_size, :]
            ValidBlock_Y = train_Y[k*block_size:(k+1)*block_size, :]
            TrainBlock_X = np.r_[train_X[:k*block_size, :], train_X[(k+1)*block_size:, :]]
            TrainBlock_Y = np.r_[train_Y[:k*block_size, :], train_Y[(k+1)*block_size:, :]]
            # mini-batch
            for Idx in range(int(TrainBlock_X.shape[0]/BATCH_SIZE)):
                # 获取索引
                StartIdx = Idx*BATCH_SIZE
                EndIdx = StartIdx+BATCH_SIZE
                if EndIdx > TrainBlock_X.shape[0]:
                    EndIdx = TrainBlock_X.shape[0]
                W = train('cross', W, TrainBlock_X[StartIdx:EndIdx,:], TrainBlock_Y[StartIdx:EndIdx,:], LAMBDA, learning_rate)
            # 计算损失值
            train_loss[-1] += CrossEntropyLoss(W, TrainBlock_X, TrainBlock_Y)
            valid_loss[-1] += CrossEntropyLoss(W, ValidBlock_X, ValidBlock_Y)
        train_loss[-1] /= K
        valid_loss[-1] /= K
        if (i+1)%1 == 0:
            train_acc.append(accuracy(predict('cross', W, train_X), train_Y))
            test_acc.append(accuracy(predict('cross', W, test_X), test_Y))
            print("%d/%d:\tAvgTrainLoss = %.6f\tAvgValidLoss = %.6f\tTotalTrainAcc = %.6f\tTotalTestAcc = %.6f"%(i+1, epochs, train_loss[-1], valid_loss[-1], train_acc[-1], test_acc[-1]))
    time_end = time.time()
    print("time use:%f"%(time_end-time_start))

    # 设置参数矩阵
    np.random.seed(0)
    W2 = np.random.rand(train_X.shape[1], 1)
    # hinge loss模型训练
    train_loss2 = []
    valid_loss2 = []
    train_acc2 = []
    test_acc2 =[]
    time_start = time.time()
    for i in range(epochs):
        # 打乱训练集数据和标签
        state = np.random.get_state()
        np.random.shuffle(train_X)
        np.random.set_state(state)
        np.random.shuffle(train_Y)
        np.random.set_state(state)
        np.random.shuffle(train_Y2)
        # K折交叉验证
        block_size = int(train_X.shape[0]/K)
        train_loss2.append(0)
        valid_loss2.append(0)
        for k in range(K):
            ValidBlock_X = train_X[k*block_size:(k+1)*block_size, :]
            ValidBlock_Y = train_Y2[k*block_size:(k+1)*block_size, :]
            TrainBlock_X = np.r_[train_X[:k*block_size, :], train_X[(k+1)*block_size:, :]]
            TrainBlock_Y = np.r_[train_Y2[:k*block_size, :], train_Y2[(k+1)*block_size:, :]]
            # mini-batch
            for Idx in range(int(TrainBlock_X.shape[0]/BATCH_SIZE)):
                # 获取索引
                StartIdx = Idx*BATCH_SIZE
                EndIdx = StartIdx+BATCH_SIZE
                if EndIdx > TrainBlock_X.shape[0]:
                    EndIdx = TrainBlock_X.shape[0]
                W2 = train('hinge', W2, TrainBlock_X[StartIdx:EndIdx,:], TrainBlock_Y[StartIdx:EndIdx,:], LAMBDA, learning_rate)
            # 计算损失值
            train_loss2[-1] += HingeLoss(W2, TrainBlock_X, TrainBlock_Y)
            valid_loss2[-1] += HingeLoss(W2, ValidBlock_X, ValidBlock_Y)
        train_loss2[-1] /= K
        valid_loss2[-1] /= K
        if (i+1)%1 == 0:
            train_acc2.append(accuracy(predict('hinge', W2, train_X), train_Y2))
            test_acc2.append(accuracy(predict('hinge', W2, test_X), test_Y2))
            print("%d/%d:\tAvgTrainLoss = %.6f\tAvgValidLoss = %.6f\tTotalTrainAcc = %.6f\tTotalTestAcc = %.6f"%(i+1, epochs, train_loss2[-1], valid_loss2[-1], train_acc2[-1], test_acc2[-1]))
    time_end = time.time()
    print("time use:%f"%(time_end-time_start))

    # 画图
    plt.figure()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.xticks(range(0, epochs+1, 10))
    plt.plot(range(epochs), train_loss, c='red', label='train')
    plt.plot(range(epochs), valid_loss, c='green', label='valid')
    plt.legend()
    plt.grid()
    plt.savefig('CrossLoss.jpg', dpi = 900)
    plt.show()

    plt.figure()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.xticks(range(0, epochs+1, 10))
    plt.plot(range(epochs), train_loss2, c='red', label='train')
    plt.plot(range(epochs), valid_loss2, c='green', label='valid')
    plt.legend()
    plt.grid()
    plt.savefig('HingeLoss.jpg', dpi = 900)
    plt.show()

    plt.figure()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.xticks(range(0, epochs+1, 10))
    plt.plot(range(epochs), train_acc, label='Cross Train')
    plt.plot(range(epochs), train_acc2, label='Hinge Train')
    plt.plot(range(epochs), test_acc, label=' Cross Test')
    plt.plot(range(epochs), test_acc2, label=' Hinge Test')
    plt.legend()
    plt.grid()
    plt.savefig('acc.jpg', dpi = 900)
    plt.show()

if __name__=='__main__':
    import numpy as np
    import time
    from matplotlib import pyplot as plt
    main()