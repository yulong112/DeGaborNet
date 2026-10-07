import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.io as sio
import os
import h5py


## 加载数据集的函数
def loadData(name):
    # data_path = os.path.join(os.getcwd(), './dataset')
    data_path = os.path.join(os.getcwd(), '../z_Gabor_Pavia/dataset')
    if name == 'IndianPines':#145*145*200    class：16   [23, 200, 200, 100, 200, 200, 14, 200, 10, 200, 250, 200, 100, 200, 150, 45]
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'Salinas':#512*217*204  class：16
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PaviaU':#610*340*103  class:9
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'PaviaC':  # 1096*715*102  class:9
        data = sio.loadmat(os.path.join(data_path, 'Pavia.mat'))['pavia']
        labels = sio.loadmat(os.path.join(data_path, 'Pavia_gt.mat'))['pavia_gt']
    elif name == 'WHU_Hi_LongKou':#550*400*270  class：9
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
    elif name == 'WHU_Hi_HanChuan':#1217*303*274 class：16，maximum：400
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_gt.mat'))['WHU_Hi_HanChuan_gt']
    elif name == 'WHU_Hi_HongHu':#940*475*270  class：22
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu.mat'))['WHU_Hi_HongHu']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu_gt.mat'))['WHU_Hi_HongHu_gt']
    elif name == 'Houston2013':#349*1905*144  class：15
        data = sio.loadmat(os.path.join(data_path, '2013_IEEE_GRSS_DF_Contest_CASI_349_1905_144.mat'))['ans']
        labels = sio.loadmat(os.path.join(data_path, 'GRSS2013_gt.mat'))['name']
    elif name == 'Houston2018':  # 601*2384*50  class：20
        data = sio.loadmat(os.path.join(data_path, 'HoustonU_cropped.mat'))['houstonU_cropped']
        labels = h5py.File(os.path.join(data_path, 'HoustonU_gt.mat'))['houstonU_gt']
        labels = np.transpose(labels, (1, 0))
    elif name == 'KSC':#512*614*176  class：13
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    elif name == 'Btw':#1476*256*145  class：14
        data = sio.loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
        labels = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
    elif name == 'Chikusei':#2517*2335*128  class：19
        data = h5py.File(os.path.join(data_path, 'HyperspecVNIR_Chikusei_20140729.mat'))['chikusei']
        data = np.transpose(data, (2, 1, 0))
        labels = sio.loadmat(os.path.join(data_path, 'matlab.mat'))['gt']
        # labels = np.transpose(labels, (1, 0))
    elif name == 'rit':#9393*5642*7  class：19
        data = sio.loadmat(os.path.join(data_path, 'rit.mat'))['train_data']
        data = np.transpose(data, (1, 2, 0))
        labels = sio.loadmat(os.path.join(data_path, 'rit.mat'))['train_labels']
    elif name == 'MatiwanVillage':#1580*3750*256  class：20
        data = h5py.File(os.path.join(data_path, 'xiongan.mat'))['XiongAn']
        data = np.transpose(data, (2, 1, 0))
        labels = h5py.File(os.path.join(data_path, 'xiongan_gt.mat'))['xiongan_gt']
        labels = np.transpose(labels, (1, 0))
    return data, labels


def normalization(X):
    """
    Normalization: make data values within [-1,1]
    """

    img_add = X.astype('float32')
    # 对每一光谱维的数据进行操作：归一化
    # for i in range(X.shape[2]):  # X.shape[2]=args.n_channels
    #     img_add[:, :, i] = img_add[:, :, i] - img_add[:, :, i].min()
    #     img_add[:, :, i] = img_add[:, :, i] / img_add[:, :, i].max()
    #     img_add[:, :, i] = img_add[:, :, i] * 2 - 1
    # 对每一光谱维的数据进行操作：归一化
    for i in range(X.shape[2]):  # X.shape[2]=args.n_channels
        img_add[:, :, i] = (img_add[:, :, i] - img_add[:, :, i].min())/(img_add[:, :, i].max()-img_add[:, :, i].min())
        img_add[:, :, i] = img_add[:, :, i] * 2 - 1
    # 执行标准化
    # for i in range(X.shape[2]):
    #     channel_data = img_add[:, :, i]  # 选择一个通道的数据
    #     mean = np.mean(channel_data)
    #     std = np.std(channel_data)
    #     img_add[:, :, i] = (channel_data - mean) / std  # 标准化
    return img_add
    # end of normalization


## padding操作：对单个像素周围提取 patch 时，边缘像素就无法取了，因此给这部分像素进行padding操作
def padWithZeros(X, margin=7):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))  ##填充四个像素，因为此时窗口大小为15
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X  ##填充数据，nexX[：]进行切片
    return newX


## patch操作：
def createImageCubes(X, y, windowSize=15, removeZeroLabels=True):
    # 给X做padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)

    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))  ##规定数据样本、标签形状
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0  # 编号
    for r in range(margin, zeroPaddedX.shape[0] - margin):  ##进行数据填充
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:  ##清理数据
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def createImageCubes_zeropre(X, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)

    patchesData = []
    patchesLabels = []

    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1,
                    c - margin:c + margin + 1]
            label = y[r - margin, c - margin]

            if not removeZeroLabels or label > 0:
                patchesData.append(patch)
                patchesLabels.append(label)

    patchesData = np.array(patchesData)
    patchesLabels = np.array(patchesLabels)

    if removeZeroLabels:
        patchesLabels -= 1

    return patchesData, patchesLabels


def minibatcher(inputs, targets, batchsize, shuffle=False):
    """
   Generate batches.

   :param inputs: features
   :param targets: labels
   :param batchsize: the size of each batch
   :param shuffle: whether shuffle the batch.
   """

    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
    # end of minibatcher


def Sample_drawn_fixnum(n_classes, n_perclass, n_channels, feature, labels):
    """
       按固定数目生成训练集、测试集.例如每类取100个

       :param n_classes: number of predefined classes
       :param n_perclass: number of training samples per class
       :param n_channels: number of bands
       :param feature: 特征.(10249, 15, 15, 200)
       :param labels: 标签.(10249,)
       """

    samples_per_class = n_perclass  # 每一类取samples_per_class个样本，16类，即共samples_per_class*16 训练样本
    num_class = n_classes  # num_class表示类别数
    train_samples = samples_per_class * num_class  # 训练样本个数
    train_f = np.zeros((train_samples, 15, 15, n_channels))  # train_f=（n*15*15*200）
    train_labels = np.zeros(train_samples)  # train_labels=（n，1）
    test_f = np.zeros((labels.shape[0] - train_samples, 15, 15, n_channels))  # test_f（）
    test_labels = np.zeros(labels.shape[0] - train_samples)  # test_labels（）

    p = 0
    q = 0
    for i in range(num_class):  # num_class=16,i=[0-15]
        index = np.reshape(np.array(np.where(np.reshape(labels, (-1)) == i)),
                           (-1))  # 将矩阵labels变为行数为（labels=10249）行  #index为：标签为i的位置，比如labels中有哪些样本标签为1，返回那些样本的位置
        # print(index.shape)——(46,)(1428,)(830,)(237,)(483,)(730,)(28,)(478,)(20,)(972,)(2455,)(593,)(205,)(1265,)(386,)(93,)
        random_num = np.random.choice(index.__len__(), samples_per_class,
                                      replace=False)  # np.random.choice(a, size, replace)从数组中随机抽取元素：从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组 #index.__len__()——Return len(self)，samples_per_class=6 ##replace:True表示可以取相同数字，False表示不可以取相同数字
        # 例如此时样本标签0，有46个样本标签为0，index.__len__()=46，np.random.choice(46，6, replace=False)从[0，46）中输出6个数字并组成一维数组，这个值可以看作【索引】
        # print(random_num)——[ 0 20 34  4  2 23]...
        for j in range(index.__len__()):
            if np.reshape(np.array(np.where(random_num == j)), (-1)).shape[
                0] == 0:  # a.shape[0]表示矩阵的行数 #此时标签范围为0-15，j为样本i的长度 #np.where若没有符合条件的返回空
                test_f[p] = feature[index[j]]  # 测试集
                test_labels[p] = labels[index[j]]
                p += 1
            else:  # 训练集
                # print(np.where(random_num == j))——(array([3]),)...
                train_f[q] = feature[index[j]]
                train_labels[q] = labels[index[j]]
                q += 1
    return train_f, train_labels, test_f, test_labels


def Sample_drawn_fixnum_list(n_classes, n_perclass, n_channels, feature, labels):
    """
       按固定数目生成训练集、测试集.例如每类取不一样的样本数量

       :param n_classes: number of predefined classes
       :param n_perclass: [array] the fixed numbers of training samples for each class
       :param n_channels: number of bands
       :param feature: 特征.(10249, 15, 15, 200)
       :param labels: 标签.(10249,)
       """

    samples_per_class = n_perclass  # 每一类取samples_per_class个样本，16类，即共samples_per_class*16 训练样本
    num_class = n_classes  # num_class表示类别数

    train_samples = 0
    for k in range(0, len(samples_per_class)):
        train_samples += samples_per_class[k]
    # train_samples = samples_per_class * num_class  # 训练样本个数
    train_f = np.zeros((train_samples, 15, 15, n_channels))  # train_f=（n*15*15*200）
    train_labels = np.zeros(train_samples)  # train_labels=（n，1）
    test_f = np.zeros((labels.shape[0] - train_samples, 15, 15, n_channels))  # test_f（）
    test_labels = np.zeros(labels.shape[0] - train_samples)  # test_labels（）

    p = 0
    q = 0
    for i in range(num_class):
        # fetch all the samples belonging to class i
        index = np.reshape(np.array(np.where(np.reshape(labels, (-1)) == i)),
                           (-1))  # 将矩阵labels变为行数为（labels=10249）行  #index为：标签为i的位置，比如labels中有哪些样本标签为1，返回那些样本的位置
        # print(index.shape)——(46,)(1428,)(830,)(237,)(483,)(730,)(28,)(478,)(20,)(972,)(2455,)(593,)(205,)(1265,)(386,)(93,)
        random_num = np.random.choice(index.__len__(), samples_per_class[i],
                                      replace=False)  # np.random.choice(a, size, replace)从数组中随机抽取元素：从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组 #index.__len__()——Return len(self)，samples_per_class=6 ##replace:True表示可以取相同数字，False表示不可以取相同数字
        # 例如此时样本标签0，有46个样本标签为0，index.__len__()=46，np.random.choice(46，6, replace=False)从[0，46）中输出6个数字并组成一维数组，这个值可以看作【索引】
        # print(random_num)——[ 0 20 34  4  2 23]...
        for j in range(index.__len__()):
            if np.reshape(np.array(np.where(random_num == j)), (-1)).shape[
                0] == 0:  # a.shape[0]表示矩阵的行数 #此时标签范围为0-15，j为样本i的长度 #np.where若没有符合条件的返回空
                test_f[p] = feature[index[j]]  # 测试集
                test_labels[p] = labels[index[j]]
                p += 1
            else:  # 训练集
                # print(np.where(random_num == j))——(array([3]),)...
                train_f[q] = feature[index[j]]
                train_labels[q] = labels[index[j]]
                q += 1
    return train_f, train_labels, test_f, test_labels


# PCA降维
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX,numComponents
