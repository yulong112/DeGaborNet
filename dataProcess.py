import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

"""
code reference: https://blog.csdn.net/hahahd3/article/details/112199724
"""

def coordinates_to_mask(test_index, img_shape):
    """
    将坐标字典转换为mask

    Args:
        test_index: dict, {类别: (行坐标数组, 列坐标数组)}
        img_shape: tuple, (H, W)

    Returns:
        mask: np.ndarray, shape (H, W), 每个像素值为类别标签，背景为-1
    """
    H, W = img_shape
    mask = np.zeros((H, W), dtype=bool)  # 初始化为False（背景）

    for label, (rows, cols) in test_index.items():
        # 将所有有标签的像素设为True
        mask[rows, cols] = True

    return mask


def loadData(path_image='dataset/Indian-pines/Indian_pines_corrected.mat',
             path_label='dataset/Indian-pines/Indian_pines_gt.mat',
             key_image='indian_pines_corrected',
             key_label='indian_pines_gt',
             use_pca=False, pca_components=5, normalize=False):  # h, w, c
    # mat = loadmat(path_image)
    # print(mat.keys())
    features = loadmat(path_image)[key_image]

    # mat_labels = loadmat(path_label)
    # print(mat_labels.keys())
    labels = loadmat(path_label)[key_label]
    if use_pca:
        features = applyPCA(features, numComponents=pca_components)
    if normalize:
        min_val = np.min(features, axis=(0, 1))
        max_val = np.max(features, axis=(0, 1))
        features = (features - min_val) / (max_val - min_val) * 2. - 1.
    return features.astype(np.float32), labels


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX.astype(np.float32)


def zeroPadding(X: np.ndarray, margin=2):
    """
    channel_last
    :param X:
    :param margin:
    :return:
    """
    shapeX = X.shape
    newX = np.zeros((shapeX[0] + 2 * margin, shapeX[1] + 2 * margin, shapeX[2]), dtype=X.dtype)
    newX[margin: shapeX[0] + margin, margin: shapeX[1] + margin, :] = X #将输入数据复制到新数组的中心位置
    return newX


def window_slides(features: np.ndarray, labels: np.ndarray, window_size=9, removeZeroLabels=True):
    """
    channel_last
    :param removeZeroLabels:
    :param features:
    :param labels:
    :param window_size:
    :return:
    """
    h, w = labels.shape
    margin = (window_size - 1) // 2
    features = zeroPadding(X=features, margin=margin)
    patch_fea = np.zeros(shape=(h * w, window_size, window_size, features.shape[-1]), dtype=features.dtype)
    for i in range(h):
        for j in range(w):
            patch_fea[i * w + j, :, :, :] = features[i: i + margin * 2 + 1, j: j + margin * 2 + 1, :]
    patch_labels = labels.reshape((h * w,))
    if removeZeroLabels:
        patch_fea = patch_fea[patch_labels > 0, :, :, :]
        patch_labels = patch_labels[patch_labels > 0]
        patch_labels -= 1
    return patch_fea, patch_labels


def randomSplitTrainTestSet(X, y, testRatio=0.25, random_state=256):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=random_state,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def splitTrainTestSet(X: np.numarray, y: np.numarray, trainNumList: list = None, normalize=False, return_index=False):
    if trainNumList is None:
        trainNumList = [33, 50, 50, 50, 50, 50, 20, 50, 14, 50, 50, 50, 50, 50, 50, 50]  # 50
        # [33, 100, 100, 100, 100, 100, 20, 100, 14, 100, 100, 100, 100, 100, 100, 75]  # 100
        # [33, 200, 200, 181, 200, 200, 20, 200, 14, 200, 200, 200, 143, 200, 200, 75]  # 200
    trainNum = sum(trainNumList)
    total = X.shape[0]
    print('trainNum:', trainNum, 'testNum:', total - trainNum)
    X_train = np.zeros(shape=(trainNum,) + X.shape[1:], dtype=X.dtype)
    X_test = np.zeros(shape=(total - trainNum,) + X.shape[1:], dtype=X.dtype)
    y_train = np.zeros(shape=(trainNum,) + y.shape[1:], dtype=y.dtype)
    y_test = np.zeros(shape=(total - trainNum,) + y.shape[1:], dtype=y.dtype)
    item_train = 0
    item_test = 0
    tmp_train = []
    tmp_test = []
    for i in range(len(trainNumList)):
        index = np.where(y == i)
        if isinstance(index, tuple):
            index = index[0]
        index_train = np.random.choice(index, size=trainNumList[i], replace=False)
        index_test = np.setdiff1d(index, index_train)
        tmp_train.append(index_train)
        tmp_test.append(index_test)
        X_train[item_train: item_train + trainNumList[i]] = X[index_train]
        X_test[item_test: item_test + index_test.shape[0]] = X[index_test]
        y_train[item_train: item_train + trainNumList[i]] = y[index_train]
        y_test[item_test: item_test + index_test.shape[0]] = y[index_test]
        item_train += trainNumList[i]
        item_test += index_test.shape[0]
    if normalize:
        min_val = np.min(X_train, axis=(0, 1, 2))
        max_val = np.max(X_train, axis=(0, 1, 2))
        X_train = (X_train - min_val) / (max_val - min_val) * 2. - 1.
        X_test = (X_test - min_val) / (max_val - min_val) * 2. - 1.
    if return_index:
        return X_train, X_test, y_train, y_test, tmp_train, tmp_test
    else:
        return X_train, X_test, y_train, y_test


def add_gaussian_noise(HyperCube, nSig=0.01):
    """
    为高光谱数据添加高斯噪声

    Parameters:
    -----------
    HyperCube : numpy.ndarray
        原始高光谱数据，形状为 (Height, Width, Bands)
    nSig : float
        噪声水平系数

    Returns:
    --------
    HyperCube_noisy : numpy.ndarray
        添加噪声后的高光谱数据
    """
    # 1. 获取数据范围
    data_min = np.min(HyperCube)
    data_max = np.max(HyperCube)
    data_range = data_max - data_min

    if data_range == 0:
        raise ValueError("数据范围为零，无法进行归一化")

    # 2. 全局归一化到 [0, 1]（所有波段使用相同的缩放）
    OriData3_normalized = (HyperCube - data_min) / data_range

    # 3. 获取维度
    Height, Width, Bands = OriData3_normalized.shape

    # 4. 添加高斯噪声（在归一化空间）
    noiselevel = nSig * np.ones(Bands)
    oriData3_noise = OriData3_normalized.copy()

    for i in range(Bands):
        np.random.seed(i)
        noise = noiselevel[i] * np.random.randn(Height, Width)
        oriData3_noise[:, :, i] = OriData3_normalized[:, :, i] + noise

    # 5. 还原到原始数据范围
    HyperCube_noisy = oriData3_noise * data_range + data_min

    return HyperCube_noisy


if __name__ == '__main__':
    a, b = loadData(path_image='../dataset/Indian-pines/Indian_pines_corrected.mat',
                    path_label='../dataset/Indian-pines/Indian_pines_gt.mat',
                    key_image='indian_pines_corrected',
                    key_label='indian_pines_gt')
    print(np.unique(a), np.unique(b))
    print(a.dtype, b.dtype)
    print(a.shape, b.shape)
