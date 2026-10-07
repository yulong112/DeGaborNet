"""
@Time    : 2022/10/30 15:15
@Author  : Lin Luo
@FileName: lightSpectrumDataset.py
@describe TODO
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from zdataset import *
from dataProcess import zeroPadding


def splitTrainTestIndex(labels: np.ndarray, trainNumList: list = None, removeZeroLabels=True):
    """
    :param removeZeroLabels:
    :param labels: h*w
    :param trainNumList:
    :return:
    """
    train_index = {}
    test_index = {}
    if removeZeroLabels:
        labels_tmp = labels.copy().astype(int) - 1  # 标签矩阵,去除未标记的元素，将标签由1变为0开始
    else:
        labels_tmp = labels.copy()
    keys = np.unique(labels_tmp)
    for key in keys:  # -1, 0,1,2,3,,,15
        if key == -1 and removeZeroLabels:
            continue
        index_h, index_w = np.where(labels_tmp == key)  # 返回第一维,第二维的索引
        # index_h, index_w = (46,) 第一类
        total_ids = np.arange(index_h.shape[0])  # 从零开始  # 该标签共有多少样本##认为是将当前类别所有的横坐标进行汇总，存储的行序号
        train_ids = np.random.choice(total_ids, size=trainNumList[key], replace=False)  # False表示不可以取相同数字#用于构建训练集数量
        test_ids = np.setdiff1d(total_ids, train_ids)  # setdiff1d的作用是求两个数组的集合差#用于将除训练集之外的数据作为测试集
        train_index[key] = (index_h[train_ids], index_w[#获取训练集的行坐标与列坐标
            train_ids])  # train_index[key]里有两个元组  # train_index是字典  # train_index[key]是元组 # 元组里是ndarray:(33,)
        test_index[key] = (index_h[test_ids], index_w[test_ids])#获取测试集的行坐标与列坐标
    return train_index, test_index


class LightSpectrumDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, index: dict, window_size=15, use_lstm=False, dim=2,
                 removeZeroLabels=True, normalize=False):
        """
        channel_last
        :param features:
        :param labels:
        :param index: {1: (h, w), ...}
        """
        super(LightSpectrumDataset, self).__init__()
        if normalize:  # 是否归一化
            min_val = np.min(features, axis=(0, 1))
            max_val = np.max(features, axis=(0, 1))
            features = (features - min_val) / (max_val - min_val) * 2. - 1.
        self.height, self.width, _ = features.shape

        self.index = []
        for key in index.keys():
            index_h, index_w = index[key]#将每一个类的行坐标与列坐标分别赋值给index_h,index_w
            for item in range(len(index_h)):
                self.index.append([index_h[item], index_w[item]])         # 最后list{800} 16x50
        self.dim = dim
        self.use_lstm = use_lstm
        if removeZeroLabels:                                               # 是否移除0标签
            labels = labels.copy().astype(np.integer) - 1
        self.y = np.zeros(shape=(len(self.index), 1), dtype=np.integer)
        for item in range(len(self.index)):
            self.y[item] = labels[self.index[item][0], self.index[item][1]]
        self.labels = torch.LongTensor(labels)
        self.margin = (window_size - 1) // 2
        self.features = zeroPadding(features, margin=self.margin)            # padding图像
        # 改变形状（提取patch块在后面）
        '''
        if self.use_lstm:
            self.features = self.features.transpose(2, 0, 1)
            if self.dim == 2:
                self.features = self.features[:, np.newaxis, ...]
            elif self.dim == 3:
                self.features = self.features[np.newaxis, np.newaxis, ...]
            else:
                raise ValueError(f'use_lstm={use_lstm} & dim={dim}...')
        else:
        '''
        self.features = self.features.transpose(2, 0, 1)#由0,1,2变换2,0,1
        if self.dim == 3:
            self.features = self.features[np.newaxis, ...]
        elif self.dim != 2:
            raise ValueError(f'use_lstm={use_lstm} & dim={dim}...')

    def getY(self):
        return self.y

    def __getitem__(self, item):
        coor_h, coor_w = self.index[item]
        # print(coor_h, coor_w, '-----------', self.labels[coor_h, coor_w])
        '''
        if self.use_lstm:
            if self.dim == 2:
                fea = self.features[:, :, coor_h: coor_h + self.margin * 2 + 1,
                      coor_w: coor_w + self.margin * 2 + 1].astype(np.float32)
            elif self.dim == 3:
                fea = self.features[:, :, :, coor_h: coor_h + self.margin * 2 + 1,
                      coor_w: coor_w + self.margin * 2 + 1].astype(np.float32)
            else:
                raise ValueError(f'use_lstm={self.use_lstm} & dim={self.dim}...')
        else:
        '''
        if self.dim == 2:
            fea = self.features[:, coor_h: coor_h + self.margin * 2 + 1,
                  coor_w: coor_w + self.margin * 2 + 1].astype(np.float32)
        elif self.dim == 3:
            fea = self.features[:, :, coor_h: coor_h + self.margin * 2 + 1,
                  coor_w: coor_w + self.margin * 2 + 1].astype(np.float32)
        else:
            raise ValueError(f'use_lstm={self.use_lstm} & dim={self.dim}...')
        return torch.FloatTensor(fea), self.labels[coor_h, coor_w]

    def __len__(self):
        return len(self.index)


def getpredictindex(labels: np.ndarray):
    """
    获取全部元素位置index
    :param labels: 标签矩阵 h*w
    :return:
    """
    pre_index = {}
    labels_tmp = labels.copy()
    keys = np.unique(labels_tmp)  # 0-16
    for key in keys:
        index_h, index_w = np.where(labels_tmp == key)
        total_ids = np.arange(index_h.shape[0])
        pre_index[key] = (index_h[total_ids], index_w[total_ids])
    return pre_index


if __name__ == '__main__':
    X, y = loadData('IP')
    pre_index = getpredictindex(y)

    tmp = []
    a = np.random.rand(100, 100, 200)
    for ulstm in [False, True]:
        for dimm in [2, 3]:
            b = np.random.randint(low=0, high=10, size=(100, 100), dtype=np.int8)
            print(np.unique(b))
            train_index, test_index = splitTrainTestIndex(b, trainNumList=[10 for _ in range(9)], removeZeroLabels=True)
            m = LightSpectrumDataset(a, b, use_lstm=ulstm, dim=dimm, index=train_index, window_size=9)
            tmp.append(len(m))
            for i in range(len(m)):
                c = m[i]
                print(i + 1, ulstm, dimm, len(m), c[0].shape, c[1])
    print(tmp)
