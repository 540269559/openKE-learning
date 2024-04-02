#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PyTorchTrainDataset(Dataset):
    # 初始化函数，用于设置数据集的头实体、尾实体、关系、实体总数、
    # 关系总数等属性，并设置采样模式、负例实体数、负例关系数等参数。
    def __init__(self, head, tail, rel, ent_total, 
                rel_total, sampling_mode = 'normal', 
                bern_flag = False, filter_flag = True, 
                neg_ent = 1, neg_rel = 0):
        # 头实体、尾实体、实体关系
        self.head = head
        self.tail = tail
        self.rel = rel
        # 实体、关系、三元组的总数
        self.rel_total = rel_total
        self.ent_total = ent_total
        self.tri_total = len(head)
        # 抽样模式
        self.sampling_mode = sampling_mode
        # 负样本样例
        self.neg_ent = neg_ent
        self.neg_rel = neg_rel
        self.bern_flag = bern_flag
        self.filter_flag = filter_flag
        if self.sampling_mode == "normal":
            self.cross_sampling_flag = None
        else:
            self.cross_sampling_flag = 0
        self.__count_htr()

    def __len__(self):
        return self.tri_total

    def __getitem__(self, idx):
        return (self.head[idx], self.tail[idx], self.rel[idx])    
    
    def collate_fn(self, data):
        batch_data = {}
        # 正常模式
        if self.sampling_mode == "normal":
            batch_data['mode'] = "normal"
            # 将输入数据中的头实体、尾实体和关系提取出来，并对其进行重塑。
            # 通过 np.repeat 函数，将每个实体和关系重复若干次，以便与正负例对齐。
            batch_h = np.array([item[0] for item in data]).reshape(-1, 1)
            batch_t = np.array([item[1] for item in data]).reshape(-1, 1)
            batch_r = np.array([item[2] for item in data]).reshape(-1, 1)
            batch_h = np.repeat(batch_h, 1 + self.neg_ent + self.neg_rel, axis = -1)
            batch_t = np.repeat(batch_t, 1 + self.neg_ent + self.neg_rel, axis = -1)
            batch_r = np.repeat(batch_r, 1 + self.neg_ent + self.neg_rel, axis = -1)
            for index, item in enumerate(data):
                last = 1
                # 是否需要更换实体产生负样本
                if self.neg_ent > 0:
                    neg_head, neg_tail = self.__normal_batch(item[0], item[1], item[2], self.neg_ent)
                    if len(neg_head) > 0:
                        # 将对应位置的数值更换成neg_head
                        batch_h[index][last:last + len(neg_head)] = neg_head
                        last += len(neg_head)
                    if len(neg_tail) > 0:
                        batch_t[index][last:last + len(neg_tail)] = neg_tail
                        last += len(neg_tail)
                # 是否需要更换实体关系产生负样本
                if self.neg_rel > 0:
                    neg_rel = self.__rel_batch(item[0], item[1], item[2], self.neg_rel)
                    batch_r[index][last:last + len(neg_rel)] = neg_rel
            # 将头实体、尾实体和关系的批数据进行转置，以便与 TensorFlow 的习惯一致
            # （TensorFlow 通常使用 [batch_size, sequence_length] 的形状）。
            batch_h = batch_h.transpose()
            batch_t = batch_t.transpose()
            batch_r = batch_r.transpose()
        # 选择头部采样或者尾部采样
        else:
            self.cross_sampling_flag = 1 - self.cross_sampling_flag
            if self.cross_sampling_flag == 0:
                batch_data['mode'] = "head_batch"
                batch_h = np.array([[item[0]] for item in data])
                batch_t = np.array([item[1] for item in data])
                batch_r = np.array([item[2] for item in data])
                batch_h = np.repeat(batch_h, 1 + self.neg_ent, axis = -1)
                for index, item in enumerate(data):
                    neg_head = self.__head_batch(item[0], item[1], item[2], self.neg_ent)
                    batch_h[index][1:] = neg_head
                batch_h = batch_h.transpose()
            else:
                batch_data['mode'] = "tail_batch"
                batch_h = np.array([item[0] for item in data]) 
                batch_t = np.array([[item[1]] for item in data])
                batch_r = np.array([item[2] for item in data])
                batch_t = np.repeat(batch_t, 1 + self.neg_ent, axis = -1)
                for index, item in enumerate(data):
                    neg_tail = self.__tail_batch(item[0], item[1], item[2], self.neg_ent)
                    batch_t[index][1:] = neg_tail
                batch_t = batch_t.transpose()

        # 生成正样例和负样例，正样例为1，负样例为0
        batch_y = np.concatenate([np.ones((len(data), 1)), np.zeros((len(data), self.neg_ent + self.neg_rel))], -1).transpose()
        # 将数组降维，删掉维度上只有1的维度
        batch_data['batch_h'] = batch_h.squeeze()
        batch_data['batch_t'] = batch_t.squeeze()
        batch_data['batch_r'] = batch_r.squeeze()
        batch_data['batch_y'] = batch_y.squeeze()
        return batch_data

    # 替换头部生成负样本
    def __corrupt_head(self, t, r, num_max = 1):
        # 创建一个大小为 num_max 的随机整数数组，范围是 [0, self.ent_total)，表示实体的索引。
        tmp = torch.randint(low = 0, high = self.ent_total, size = (num_max, )).numpy()
        # 如果 filter_flag 为 False，则直接返回生成的随机整数数组，不进行过滤。
        if not self.filter_flag:
            return tmp
        # 创建了一个布尔掩码数组，用于过滤已存在于训练集中的头实体。
        # np.in1d 函数用于检查数组 tmp 中的元素是否在 self.h_of_tr[(t, r)] 中存在，并返回一个布尔数组。
        # assume_unique=True 表示假设输入的数组都是唯一的，这可以加快计算速度。
        # invert=True 表示返回一个与结果相反的布尔数组，即 True 表示不存在于训练集中的头实体。
        mask = np.in1d(tmp, self.h_of_tr[(t, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    # 替换尾部生成负样本
    def __corrupt_tail(self, h, r, num_max = 1):
        tmp = torch.randint(low = 0, high = self.ent_total, size = (num_max, )).numpy()
        if not self.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.t_of_hr[(h, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    # 替换实体关系生成负样本
    def __corrupt_rel(self, h, t, num_max = 1):
        tmp = torch.randint(low = 0, high = self.rel_total, size = (num_max, )).numpy()
        if not self.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.r_of_ht[(h, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    # 这个函数主要是用来生成正例对应的负例集合，其中包括头实体和尾实体的负例。
    def __normal_batch(self, h, t, r, neg_size):
        neg_size_h = 0
        neg_size_t = 0
        # 根据 Bernoulli 采样标志 (bern_flag) 确定使用哪种采样方式。
        # 如果 bern_flag 为 True，则采用 Bernoulli 采样，否则使用均匀采样。
        # 这里计算采样概率为头实体的右平均值与左平均值之比。
        prob = self.rig_mean[r] / (self.rig_mean[r] + self.lef_mean[r]) if self.bern_flag else 0.5
        for i in range(neg_size):
            if random.random() < prob:
                neg_size_h += 1
            else:
                neg_size_t += 1

        neg_list_h = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_h:
            neg_tmp_h = self.__corrupt_head(t, r, num_max = (neg_size_h - neg_cur_size) * 2)
            neg_list_h.append(neg_tmp_h)
            neg_cur_size += len(neg_tmp_h)
        if neg_list_h != []:
            neg_list_h = np.concatenate(neg_list_h)
            
        neg_list_t = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_t:
            neg_tmp_t = self.__corrupt_tail(h, r, num_max = (neg_size_t - neg_cur_size) * 2)
            neg_list_t.append(neg_tmp_t)
            neg_cur_size += len(neg_tmp_t)
        if neg_list_t != []:
            neg_list_t = np.concatenate(neg_list_t)

        return neg_list_h[:neg_size_h], neg_list_t[:neg_size_t]

    # 用于生成负例批次，根据不同的采样模式生成不同类型的负例数据。
    def __head_batch(self, h, t, r, neg_size):
        neg_list = []
        neg_cur_size = 0
        # 开始一个循环，直到生成的负例数量达到指定的数量
        while neg_cur_size < neg_size:
            """
            neg_size - neg_cur_size 表示还需要生成的负例数量。
            为了确保在生成负例时能够达到指定数量 neg_size，我们将剩余需要生成的负例数量扩大了一倍，即乘以2。
            这样做的目的是为了避免在过程中由于重复负例的产生而导致生成数量不足。
            调用 __corrupt_head 函数生成负例头实体，参数包括尾实体 t、关系 r，以及剩余需要生成的负例数量。
            因为生成的负例可能存在重复，所以为了确保生成数量达到要求，这里将预留的生成数量扩大了一倍。
            """
            neg_tmp = self.__corrupt_head(t, r, num_max = (neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        # 将生成的负例头实体列表连接起来，并截取前 neg_size 个负例作为最终结果返回。
        return np.concatenate(neg_list)[:neg_size]

    def __tail_batch(self, h, t, r, neg_size):
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.__corrupt_tail(h, r, num_max = (neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def __rel_batch(self, h, t, r, neg_size):
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.__corrupt_rel(h, t, num_max = (neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]
    

    # set函数
    def set_sampling_mode(self, sampling_mode):
        self.sampling_mode = sampling_mode

    def set_ent_neg_rate(self, rate):
        self.neg_ent = rate

    def set_rel_neg_rate(self, rate):
        self.neg_rel = rate

    def set_bern_flag(self, bern_flag):
        self.bern_flag = bern_flag

    def set_filter_flag(self, filter_flag):
        self.filter_flag = filter_flag

    # get函数
    def get_ent_tot(self):
        return self.ent_total

    def get_rel_tot(self):
        return self.rel_total

    def get_tri_tot(self):
        return self.tri_total

class PyTorchTrainDataLoader(DataLoader):
    # 创建一个 PyTorch 的训练数据加载器对象，用于加载训练数据并准备用于模型训练。
    # drop_last:如果数据集大小不能整除批次大小，是否丢弃最后一个不完整的批次。
    # shuffle: 是否在每个 epoch 之前对数据进行随机重排。
    def __init__(self, 
        in_path = None, 
        tri_file = None,
        ent_file = None,
        rel_file = None,
        batch_size = None, 
        nbatches = None, 
        threads = 8, 
        sampling_mode = "normal", 
        bern_flag = False, 
        filter_flag = True, 
        neg_ent = 1, 
        neg_rel = 0, 
        shuffle = True, 
        drop_last = True):

        self.in_path = in_path
        self.tri_file = tri_file
        self.ent_file = ent_file
        self.rel_file = rel_file
        if in_path != None:
            self.tri_file = in_path + "train2id.txt"
            self.ent_file = in_path + "entity2id.txt"
            self.rel_file = in_path + "relation2id.txt"

        dataset = self.__construct_dataset(sampling_mode, bern_flag, filter_flag, neg_ent, neg_rel)

        self.batch_size = batch_size
        self.nbatches = nbatches
        if batch_size == None:
            self.batch_size = dataset.get_tri_tot() // nbatches
        if nbatches == None:
            self.nbatches = dataset.get_tri_tot() // batch_size

        super(PyTorchTrainDataLoader, self).__init__(
            dataset = dataset,
            batch_size = self.batch_size,
            shuffle = shuffle,
            pin_memory = True,
            num_workers = threads,
            collate_fn = dataset.collate_fn,
            drop_last = drop_last)

    # 这个函数的主要目的是根据提供的文件路径和参数创建一个 PyTorchTrainDataset 对象，即构建训练数据集对象。
    def __construct_dataset(self, sampling_mode, bern_flag, filter_flag, neg_ent, neg_rel):
        f = open(self.ent_file, "r")
        ent_total = (int)(f.readline())
        f.close()

        f = open(self.rel_file, "r")
        rel_total = (int)(f.readline())
        f.close()

        head = []
        tail = []
        rel = []

        f = open(self.tri_file, "r")
        triples_total = (int)(f.readline())
        for index in range(triples_total):
            h,t,r = f.readline().strip().split()
            head.append((int)(h))
            tail.append((int)(t))
            rel.append((int)(r))
        f.close()

        dataset = PyTorchTrainDataset(np.array(head), np.array(tail), np.array(rel), ent_total, rel_total, sampling_mode, bern_flag, filter_flag, neg_ent, neg_rel)
        return dataset

    """interfaces to set essential parameters"""

    def set_sampling_mode(self, sampling_mode):
        self.dataset.set_sampling_mode(sampling_mode)

    def set_work_threads(self, work_threads):
        self.num_workers = work_threads

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches
        self.batch_size = self.tripleTotal // self.nbatches

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.nbatches = self.tripleTotal // self.batch_size

    def set_ent_neg_rate(self, rate):
        self.dataset.set_ent_neg_rate(rate)

    def set_rel_neg_rate(self, rate):
        self.dataset.set_rel_neg_rate(rate)

    def set_bern_flag(self, bern_flag):
        self.dataset.set_bern_flag(bern_flag)

    def set_filter_flag(self, filter_flag):
        self.dataset.set_filter_flag(filter_flag)
    
    """interfaces to get essential parameters"""
    
    def get_batch_size(self):
        return self.batch_size

    def get_ent_tot(self):
        return self.dataset.get_ent_tot()

    def get_rel_tot(self):
        return self.dataset.get_rel_tot()

    def get_triple_tot(self):
        return self.dataset.get_tri_tot()