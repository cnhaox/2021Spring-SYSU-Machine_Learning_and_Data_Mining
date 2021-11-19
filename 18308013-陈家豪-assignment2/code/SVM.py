# coding: utf-8
import numpy as np
import time
from sklearn import svm

# 获取训练集数据
train_images = np.load("material/train-images.npy")
train_labels = np.load("material/train-labels.npy")
# 添加偏置1
extend = np.ones(train_images.shape[0])
train_X = np.c_[extend, train_images]
train_Y = train_labels
# 特征标准化
meanVal = []
stdVal = []
for i in range(train_X.shape[1]):
    meanVal.append(np.mean(train_X[:, i]))
    stdVal.append(np.std(train_X[:,i]))
    if stdVal[i] != 0:
        train_X[:, i] = (train_X[:,i]-meanVal[i])/stdVal[i]
# 打乱数据
state = np.random.get_state()
np.random.shuffle(train_X)
np.random.set_state(state)
np.random.shuffle(train_Y)

# 获取测试集数据
test_images = np.load("material/test-images.npy")
test_labels = np.load("material/test-labels.npy")
# 添加偏置1
extend = np.ones(test_images.shape[0])
test_X = np.c_[extend, test_images]
test_Y = test_labels
# 特征标准化
for i in range(test_X.shape[1]):
    if stdVal[i] != 0:
        test_X[:, i] = (test_X[:,i]-meanVal[i])/stdVal[i]
# 打乱数据
state = np.random.get_state()
np.random.shuffle(test_X)
np.random.set_state(state)
np.random.shuffle(test_Y)

# 线性核SVM初始化与训练
LinearSvc = svm.SVC(C=1.0, kernel='linear', random_state=0)
time_start = time.time()
model1 = LinearSvc.fit(train_X, train_Y)
time_end = time.time()
# 显示准确率
print("--------- Linear SVM ---------")
print("time use:\t%f"%(time_end-time_start))
print("train acc:\t%f"%(model1.score(train_X,train_Y)))
print("test acc:\t%f"%(model1.score(test_X,test_Y)))

# 高斯核SVM初始化与训练
RbfSvc = svm.SVC(C=1.0, kernel='rbf', random_state=0, gamma='auto')
time_start = time.time()
model2 = RbfSvc.fit(train_X, train_Y)
time_end = time.time()
# 显示准确率
print("---------- RBF SVM ----------")
print("time use:\t%f"%(time_end-time_start))
print("train:\t\t%f"%(model2.score(train_X,train_Y)))
print("test:\t\t%f"%(model2.score(test_X,test_Y)))