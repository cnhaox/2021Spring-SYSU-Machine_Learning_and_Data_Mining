import numpy as np
import random
import time
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from scipy.linalg import pinvh
from munkres import Munkres, print_matrix
EPOCHS = 20
NEW_DIMENSION = 60

class myKmeans():
    def __init__(self, init_data, K=2, init_method='random'):
        '''
        Description
        ----------
        K-Means
        
        Parameters
        ----------
        init_data : ndarray (numbers, dimensions)
            用于初始化模型参数的样本数据

        K : int
            聚类数
        
        init_method : string
            初始化方法: 'random', 'distance' or 'random+distance'
        '''
        self.K = K # 聚类数
        self.dimension = init_data.shape[1] # 样本维度
        self.centroids = None # 聚类中心
        self.initCentroids(init_data, init_method)

    def initCentroids(self, data, init_method):
        '''
        Description
        ----------
        初始化聚类中心
        
        Parameters
        ----------
        data : ndarray (numbers, dimensions)
            样本数据

        init_method : string 
            'random', 'distance' or 'random+distance'
        
        Returns
        ----------
        None
        '''
        assert len(data.shape)==2, "data must be a matrix"
        assert data.shape[0]>=self.K, "The number of data must be larger than K"
        # 获取并打乱样本数据的索引
        indexes = np.arange(data.shape[0])
        np.random.shuffle(indexes)

        if init_method=='random':
            # 随机选择K个样本数据作为聚类中心
            self.centroids = np.zeros((self.K, data.shape[1]))
            for i in range(self.K):
                self.centroids[i] = data[indexes[i]]
        elif init_method=='distance':
            # 随机选择1个样本数据作为聚类中心，
            # 然后依次选取与当前聚类中心最远的样本作为聚类中心
            self.centroids = np.zeros((1, data.shape[1]))
            usedIndexes = list()
            for i in range(self.K):
                if i==0:
                    # 随机选择一个
                    self.centroids[i] = data[indexes[i]]
                    usedIndexes.append(indexes[i])
                else:
                    # 计算其余样本离当前聚类中心的总距离
                    distances = self.getDistance(data)
                    totalDistances = np.sum(distances, axis=1)
                    # 获取从远到近的排序
                    indexes = np.argsort(-totalDistances)
                    for index in indexes:
                        if index not in usedIndexes:
                            self.centroids = np.insert(self.centroids, obj=self.centroids.shape[0], values=data[index], axis=0)
                            usedIndexes.append(index)
                            break
        elif init_method=='random+distance':
            # 随机选择1个样本数据作为聚类中心，
            # 然后依次从与当前聚类中心最远的样本中随机选取作为聚类中心
            self.centroids = np.zeros((1, data.shape[1]))
            usedFlags = np.zeros(data.shape[0], dtype=int)
            for i in range(self.K):
                if i==0:
                    # 随机选择一个
                    self.centroids[i] = data[indexes[i]]
                    usedFlags[indexes[i]] = 1
                else:
                    # 获取没选的样本索引
                    unusedIndexes = np.argsort(usedFlags)
                    unusedIndexes = unusedIndexes[:-i]
                    # 计算没选的样本离聚类中心的总距离
                    distances = self.getDistance(data[unusedIndexes,:])
                    totalDistances = np.sum(distances, axis=1)
                    # 从远到近排序
                    DistancesIndexes = np.argsort(-totalDistances)
                    while True:
                        # 随机抽号，位于前50%且没被选过时即为新聚类中心
                        index = unusedIndexes[random.randint(0,unusedIndexes.size-1)]
                        if index in unusedIndexes[DistancesIndexes[:int(np.ceil(float(DistancesIndexes.size)/2))]]:
                            usedFlags[index] = 1
                            self.centroids = np.insert(self.centroids, obj=self.centroids.shape[0], values=data[index], axis=0)
                            break
    
    def getDistance(self, data):
        '''
        Description
        ----------
        获取data与各个聚类中心的距离
        
        Parameters
        ----------
        data : ndarray (numbers, dimensions)
            样本数据
        
        Returns
        ----------
        distances : ndarray (numbers, K)
            distances[i,k]表示第i个样本与第k个聚类中心的欧式距离
        '''
        distances = np.zeros((data.shape[0], self.centroids.shape[0]))
        for i in range(data.shape[0]):
            distances[i] = np.sum((self.centroids - data[i])**2, axis=1)**0.5
        return distances 

    def getClusters(self, data):
        '''
        Description
        ----------
        获取各个样本的所属聚类，以及离所属聚类中心的平均距离
        
        Parameters
        ----------
        data : ndarray (numbers, dimensions)
            样本数据
        
        Returns
        ----------
        clusters : ndarray (numbers, )
            各个样本所属聚类
        avgDistances : ndarray (numbers, )
            各个样本离所属聚类中心的平均距离
        '''
        distances = self.getDistance(data)
        clusters = np.argmin(distances, axis=1)
        avgDistances = np.sum(np.min(distances, axis=1))/data.shape[0]
        return clusters, avgDistances
    
    def getCentroids(self, data, clusters):
        '''
        Description
        ----------
        获取新的聚类中心
        
        Parameters
        ----------
        data : ndarray (numbers, dimensions)
            样本数据
        clusters : ndarray (numbers, )
            各个样本所属聚类
        
        Returns
        ----------
        centroids : ndarray (K, dimensions)
            各个聚类中心
        '''
        oneHotClusters = np.zeros((data.shape[0], self.K))
        oneHotClusters[clusters[:,None]==np.arange(self.K)] = 1
        return np.dot(oneHotClusters.T, data)/np.sum(oneHotClusters, axis=0).reshape((-1,1))
    
    def getAccuracy(self, predLabels, trueLabels):
        '''
        Description
        ----------
        计算准确率
        
        Parameters
        ----------
        predLabels : ndarray (numbers, )
            样本的预测聚类标签
        trueLabels : ndarray (numbers, )
            样本的真实聚类标签
        
        Returns
        ----------
        accuracy : float
            准确率
        '''
        assert predLabels.size==trueLabels.size, "predLabels.size must be equal to trueLabels.size"
        # 获取标签类型
        predLabelType = np.unique(predLabels)
        trueLabelType = np.unique(trueLabels)
        # 获取标签数量
        labelNum = np.maximum(len(predLabelType), len(trueLabelType))
        # 计算代价矩阵
        costMatrix = np.zeros((labelNum, labelNum))
        for i in range(len(predLabelType)):
            chosenPredLabels = (predLabels==predLabelType[i]).astype(float)
            for j in range(len(trueLabelType)):
                chosenTrueLabels = (trueLabels==trueLabelType[j]).astype(float)
                costMatrix[i,j] = -np.sum(chosenPredLabels*chosenTrueLabels)
        # 匈牙利算法
        m = Munkres()
        # 获取标签映射
        indexes = m.compute(costMatrix)
        # 获取映射后的预测标签
        mappedPredLabels = np.zeros_like(predLabels, dtype=int)
        for index1,index2 in indexes:
            if index1<len(predLabelType) and index2<len(trueLabelType):
                mappedPredLabels[predLabels==predLabelType[index1]] = trueLabelType[index2]
        return np.sum((mappedPredLabels==trueLabels).astype(float))/trueLabels.size
    
    def train(self, data):
        '''
        Description
        ----------
        训练函数
        
        Parameters
        ----------
        data : ndarray (numbers, dimensions)
            训练集数据
        
        Returns
        ----------
        accuracy : float
            聚类中心变化的值
        '''
        clusters, _ = self.getClusters(data)
        newCentroids = self.getCentroids(data, clusters)
        diff = np.sum((newCentroids-self.centroids)**2)**0.5
        self.centroids = newCentroids
        return diff

    def test(self, data, labels):
        '''
        Description
        ----------
        测试函数
        
        Parameters
        ----------
        data : ndarray (numbers, dimensions)
            测试集数据
        
        Returns
        ----------
        accuracy : float
            准确率
        distance : float
            所有样本离其聚类中心的距离平均
        '''
        clusters, avgDistance = self.getClusters(data)
        return self.getAccuracy(clusters, labels), avgDistance

    def output(self, data):
        clusters, _ = self.getClusters(data)
        return clusters


class myGMM():
    def __init__(self, n_components, init_data, init_type='random', cov_type='full', reg_covar=1e-6, isUsedPi=False):
        '''
        Description
        ----------
        GMM
        
        Parameters
        ----------
        n_components : int
            高斯子模型数

        init_data : ndarray (numbers, dimensions)
            用于初始化模型参数的样本数据

        init_type : string
            初始化方法: 'random', 'random_resp', 'kmeans'
        
        cov_type : string
            协方差矩阵类型: 'full', 'tied', 'diag', 'spherical'
        
        reg_covar : float
            用于防止协方差矩阵不可逆而添加的微小数

        isUsedPi : bool
            是否在计算高斯函数时加入(2*\pi)^{D/2}部分（不影响结果）
        '''
        assert len(init_data.shape)==2, "The shape of init_data must be (numbers, dimension)."
        self.n_components = n_components
        self.dimension = init_data.shape[1]
        self.weights = None
        self.means = None
        self.cov = None
        self.cov_type = cov_type
        self.reg_covar = reg_covar
        self.isUsedPi = isUsedPi
        self._init_parameters(init_type=init_type, cov_type=cov_type, init_data=init_data)

    def _init_parameters(self, init_data, init_type='random', cov_type='full'):
        '''
        Description
        ----------
        初始化模型参数
        
        Parameters
        ----------
        init_data : ndarray (numbers, dimensions)
            用于初始化的样本数据

        init_type : string
            初始化方法: 'random', 'random_resp', 'kmeans'
        
        cov_type : string
            协方差矩阵类型: 'full', 'tied', 'diag', 'spherical'
        
        Return
        ----------
        None
        '''
        if init_type=='random':
            # 从样本中随机抽取样本作为高斯函数的均值向量
            indexes = np.arange(init_data.shape[0])
            np.random.shuffle(indexes)
            self.means = np.zeros((self.n_components, self.dimension))
            self.means = init_data[indexes[:self.n_components]]
            # 权重初始化为统一值
            self.weights = np.ones(self.n_components)/self.n_components

            # 使用样本数据生成协方差矩阵
            tempCov = np.cov(init_data, rowvar=False)
            tempCov += np.eye(self.dimension)*self.reg_covar
            if cov_type=='full':
                # 每个高斯子模型分别使用一个协方差矩阵
                # self.cov.shape: (self.n_components, self.dimension, self.dimension)
                self.cov = tempCov[np.newaxis,:].repeat(self.n_components, axis=0)
            elif cov_type=='tied':
                # 所有高斯子模型共用一个协方差矩阵
                # self.cov.shape: (self.dimension, self.dimension)
                self.cov = tempCov
            elif cov_type=='diag':
                # 每个高斯子模型分别使用一个不要求值一样的对角协方差矩阵
                # self.cov.shape: (self.n_components, self.dimension)
                self.cov = np.diag(tempCov)
                self.cov = self.cov[np.newaxis, :].repeat(self.n_components, axis=0)
            elif cov_type=='spherical':
                # 每个高斯子模型分别使用一个要求值一样的对角协方差矩阵
                # self.cov.shape: (self.n_components, )
                self.cov = np.ones(self.n_components)*np.diag(tempCov).mean()
                
        elif init_type=='random_resp':
            # 随机初始化\gamma值，然后通过E-step获取各个参数
            assert init_data is not None, "init_data is needed."
            gamma = np.random.rand(init_data.shape[0], self.n_components)
            gamma /= np.sum(gamma, axis=1).reshape(-1,1)
            self.MStep(init_data, gamma)
        
        elif init_type=='kmeans':
            # 通过KMeans模型获得\gamma值，然后通过E-step获取各个参数
            model = myKmeans(init_data, self.n_components)
            for i in range(5):
                model.train(init_data)
            predictLabels = model.output(init_data)
            gamma = np.eye(self.n_components)[predictLabels]
            self.MStep(init_data, gamma)
            
    def gaussianFunc(self, x, mean, cov):
        '''
        Description
        ----------
        高斯函数
        
        Parameters
        ----------
        x : ndarray (numbers, dimensions) or ndarray (dimensions, )

        mean : ndarray (dimensions, )
            高斯均值
        
        cov : ndarray (dimensions, dimensions)
            协方差矩阵
        
        Returns
        ----------
        y : ndarray (numbers, ) or float
            准确率
        distance : float
            所有样本离其聚类中心的距离平均
        '''
        dev = x - mean
        maha = np.sum(np.dot(dev, np.linalg.pinv(cov))*dev, axis=1)
        y = np.exp(-0.5*maha)/np.sqrt(np.linalg.det(cov))
        if self.isUsedPi:
            return y / (2*np.pi)**(self.dimension/2)
        else:
            return y
    
    def EStep(self, data):
        '''
        Description
        ----------
        E步
        
        Parameters
        ----------
        data : ndarray (numbers, dimensions)
            训练集数据

        Returns
        ----------
        gamma : ndarray (numbers, n_components)
            \gamma
        '''
        gamma = np.zeros((data.shape[0], self.n_components))
        tempCov = None
        # 复原协方差矩阵
        if self.cov_type=='full':
            # self.cov.shape: (self.n_components, self.dimension, self.dimension)
            tempCov = self.cov
        elif self.cov_type=='tied':
            # self.cov.shape: (self.dimension, self.dimension)
            tempCov = self.cov[np.newaxis,:].repeat(self.n_components, axis=0)
        elif self.cov_type=='diag':
            # self.cov.shape: (self.n_components, self.dimension)
            tempCov = np.zeros((self.n_components, self.dimension, self.dimension))
            for k in range(self.n_components):
                tempCov[k] = np.diag(self.cov[k])
        elif self.cov_type=='spherical':
            # self.cov.shape: (self.n_components, )
            tempCov = np.zeros((self.n_components, self.dimension, self.dimension))
            for k in range(self.n_components):
                tempCov[k] = np.eye(self.dimension)*self.cov[k]

        for k in range(self.n_components):
            gamma[:,k] = self.weights[k]*self.gaussianFunc(data, self.means[k], tempCov[k])
            # gamma[:,k] = self.weights[k]*multivariate_normal(self.means[k], tempCov[k]).pdf(data)
        gamma /= np.sum(gamma, axis=1).reshape(-1,1)
        return gamma
            
    def MStep(self, data, gamma):
        '''
        Description
        ----------
        M步
        
        Parameters
        ----------
        data : ndarray (numbers, dimensions)
            训练集数据
        
        gamma : ndarray (numbers, n_components)
            \gamma

        Returns
        ----------
        None
        '''
        self.means = np.dot(gamma.T, data)/np.sum(gamma, axis=0).reshape(-1,1)
        self.weights = np.sum(gamma, axis=0)/data.shape[0]
        if self.cov_type=='full':
            # self.cov.shape: (self.n_components, self.dimension, self.dimension)
            self.cov = np.zeros((self.n_components, self.dimension, self.dimension))
            for k in range(self.n_components):
                diff = data - self.means[k]
                self.cov[k] = np.dot(gamma[:,k]*diff.T, diff)
                self.cov[k] /= np.sum(gamma[:,k])+10*np.finfo(gamma.dtype).eps# 防止除零
                self.cov[k] += np.eye(self.dimension)*self.reg_covar # 防止不可逆
        elif self.cov_type=='tied':
            # self.cov.shape: (self.dimension, self.dimension)
            self.cov = np.dot(data.T, data) - np.dot(np.sum(gamma, axis=0)*self.means.T, self.means)
            self.cov /= np.sum(gamma)
            self.cov += np.eye(self.dimension)*self.reg_covar # 防止不可逆

        elif self.cov_type=='diag':
            # self.cov.shape: (self.n_components, self.dimension)
            self.cov = np.zeros((self.n_components, self.dimension))
            for k in range(self.n_components):
                diff = data - self.means[k]
                temp = np.dot(gamma[:,k]*diff.T, diff)
                temp /= np.sum(gamma[:,k])+10*np.finfo(gamma.dtype).eps# 防止除零
                temp += np.eye(self.dimension)*self.reg_covar # 防止不可逆
                self.cov[k] = np.diag(temp)
        elif self.cov_type=='spherical':
            # self.cov.shape: (self.n_components, )
            self.cov = np.zeros((self.n_components))
            for k in range(self.n_components):
                diff = data - self.means[k]
                temp = np.dot(gamma[:,k]*diff.T, diff)
                temp /= np.sum(gamma[:,k])+10*np.finfo(gamma.dtype).eps# 防止除零
                temp += np.eye(self.dimension)*self.reg_covar # 防止不可逆
                self.cov[k] = np.diag(temp).mean()

    
    def getAccuracy(self, predLabels, trueLabels):
        '''
        Description
        ----------
        计算准确率
        
        Parameters
        ----------
        predLabels : ndarray (numbers, )
            样本的预测聚类标签
        trueLabels : ndarray (numbers, )
            样本的真实聚类标签
        
        Returns
        ----------
        accuracy : float
            准确率
        '''
        assert predLabels.size==trueLabels.size, "predLabels.size must be equal to trueLabels.size"
        # 获取标签类型
        predLabelType = np.unique(predLabels)
        trueLabelType = np.unique(trueLabels)
        # 获取标签数量
        labelNum = np.maximum(len(predLabelType), len(trueLabelType))
        # 计算代价矩阵
        costMatrix = np.zeros((labelNum, labelNum))
        for i in range(len(predLabelType)):
            chosenPredLabels = (predLabels==predLabelType[i]).astype(float)
            for j in range(len(trueLabelType)):
                chosenTrueLabels = (trueLabels==trueLabelType[j]).astype(float)
                costMatrix[i,j] = -np.sum(chosenPredLabels*chosenTrueLabels)
        # 匈牙利算法
        m = Munkres()
        # 获取标签映射
        indexes = m.compute(costMatrix)
        # 获取映射后的预测标签
        mappedPredLabels = np.zeros_like(predLabels, dtype=int)
        for index1,index2 in indexes:
            if index1<len(predLabelType) and index2<len(trueLabelType):
                mappedPredLabels[predLabels==predLabelType[index1]] = trueLabelType[index2]
        return np.sum((mappedPredLabels==trueLabels).astype(float))/trueLabels.size
    
    def train(self, data):
        '''
        Description
        ----------
        训练函数
        
        Parameters
        ----------
        data : ndarray (numbers, dimensions)
            训练集数据
        
        Returns
        ----------
        None
        '''
        gamma = self.EStep(data)
        self.MStep(data, gamma)
    
    def test(self, data, labels):
        '''
        Description
        ----------
        测试函数
        
        Parameters
        ----------
        data : ndarray (numbers, dimensions)
            测试集数据
        
        Returns
        ----------
        accuracy : float
            准确率
        loss : float
            对数似然均值
        '''
        gamma = self.EStep(data)
        loss = self.getLoss(data, gamma)
        clusters = np.argmax(gamma, axis=1)
        return self.getAccuracy(clusters, labels), loss

    def getLoss(self, data, gamma):
        '''
        Description
        ----------
        计算对数似然均值
        
        Parameters
        ----------
        data : ndarray (numbers, dimensions)
            样本数据

        gamma : ndarray (numbers, n_components)
        
        Returns
        ----------
        loss : float
            对数似然均值
        '''
        tempCov = None
        if self.cov_type=='full':
            # self.cov.shape: (self.n_components, self.dimension, self.dimension)
            tempCov = self.cov
        elif self.cov_type=='tied':
            # self.cov.shape: (self.dimension, self.dimension)
            tempCov = self.cov[np.newaxis,:].repeat(self.n_components, axis=0)
        elif self.cov_type=='diag':
            # self.cov.shape: (self.n_components, self.dimension)
            tempCov = np.zeros((self.n_components, self.dimension, self.dimension))
            for k in range(self.n_components):
                tempCov[k] = np.diag(self.cov[k])
        elif self.cov_type=='spherical':
            # self.cov.shape: (self.n_components, )
            tempCov = np.zeros((self.n_components, self.dimension, self.dimension))
            for k in range(self.n_components):
                tempCov[k] = np.eye(self.dimension)*self.cov[k]

        loss = 0
        for k in range(self.n_components):
            dev = data - self.means[k]
            loss += -0.5*np.sum(np.dot(gamma[:,k].reshape(-1,1)*dev, np.linalg.pinv(tempCov[k]))*dev)
            loss += -0.5*np.sum(gamma[:,k]*np.log(np.linalg.det(tempCov[k])))
            loss += np.sum(gamma[:,k]*np.log(self.weights[k]))
        if self.isUsedPi:
            loss += self.n_components*data.shape[0]*(-(self.dimension/2)*np.log(2*np.pi))
            return -loss/data.shape[0]
        else:
            return -loss/data.shape[0]

def main():
    np.random.seed(0)
    # 获取数据
    train_images = np.load("material/train-images.npy")
    train_labels = np.load("material/train-labels.npy")
    test_images = np.load("material/test-images.npy")
    test_labels = np.load("material/test-labels.npy")
    train_images = train_images.astype(float)
    test_images = test_images.astype(float)

    # 数据标准化
    #meanVal = []
    #stdVal = []
    #for i in range(train_images.shape[1]):
    #    meanVal.append(np.mean(train_images[:, i]))
    #    stdVal.append(np.std(train_images[:,i]))
    #    if stdVal[i] != 0:
    #        train_images[:, i] = (train_images[:,i]-meanVal[i])/stdVal[i]
    #for i in range(test_images.shape[1]):
    #    if stdVal[i] != 0:
    #        test_images[:, i] = (test_images[:,i]-meanVal[i])/stdVal[i]

    # 数据降维
    pcaModel = PCA(n_components=NEW_DIMENSION)
    pcaModel.fit(train_images)
    newTrainData = pcaModel.transform(train_images)
    newTestData = pcaModel.transform(test_images)

    #划分验证集
    state = np.random.get_state()
    np.random.shuffle(newTrainData)
    np.random.set_state(state)
    np.random.shuffle(train_labels)
    newValidData = newTrainData[int(newTrainData.shape[0]*0.8):]
    newTrainData = newTrainData[:int(newTrainData.shape[0]*0.8)]
    valid_labels = train_labels[int(train_labels.shape[0]*0.8):]
    train_labels = train_labels[:int(train_labels.shape[0]*0.8)]

    # K-Means模型
    np.random.seed(0)
    kmeansModel = myKmeans(init_data=newTrainData, K=10, init_method='random')
    start_time = time.time()
    for i in range(EPOCHS):
        diff = kmeansModel.train(newTrainData)
        print('{}/{}:\tdiff = {:.4f}'.format(i+1, EPOCHS, diff))
        acc, distance = kmeansModel.test(newTrainData, train_labels)
        print('\ttrain:\t loss = {:.4f}\t acc = {:.4f}'.format(distance, acc))
        acc, distance = kmeansModel.test(newValidData, valid_labels)
        print('\tvalid:\t loss = {:.4f}\t acc = {:.4f}'.format(distance, acc))
        acc, distance = kmeansModel.test(newTestData, test_labels)
        print('\t test:\t loss = {:.4f}\t acc = {:.4f}'.format(distance, acc))
        if diff < 1e-6:
            break
    end_time = time.time()
    print('Finish. The total time is {:.2f}s.\n'.format(end_time-start_time))

    # GMM模型
    np.random.seed(0)
    gmmModel = myGMM(n_components=10, init_data=newTrainData, init_type='kmeans',cov_type='full', isUsedPi=False)
    start_time = time.time()
    for i in range(EPOCHS):
        gmmModel.train(newTrainData)
        trainAcc, trainLoss = gmmModel.test(newTrainData, train_labels)
        validAcc, validLoss = gmmModel.test(newValidData, valid_labels)
        testAcc, testLoss = gmmModel.test(newTestData, test_labels)
        print('{}/{}:'.format(i+1, EPOCHS))
        print('\t train: loss = {:.4f}\t acc = {:.4f}'.format(trainLoss, trainAcc))
        print('\t valid: loss = {:.4f}\t acc = {:.4f}'.format(validLoss, validAcc))
        print('\t  test: loss = {:.4f}\t acc = {:.4f}'.format(testLoss, testAcc))
    end_time = time.time()
    print('Finish. The total time is {:.2f}s.\n'.format(end_time-start_time))


if __name__=='__main__':
    main()