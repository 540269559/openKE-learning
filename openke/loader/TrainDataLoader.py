# coding:utf-8
import os
import ctypes
import numpy as np
"""
它主要用于生成训练数据的迭代器。训练数据采样器接受两个参数：
1.nbatches表示要生成的批次数量
2.data_sampler表示用于生成训练数据的数据采样器。
"""
class TrainDataSampler(object):

    def __init__(self, nbatches, data_sampler):
        self.nbatches = nbatches
        self.data_sampler = data_sampler
        self.batch = 0

    # 返回自身，用于支持迭代器协议
    def __iter__(self):
        return self

    # 生成下一个批次的训练数据
    def __next__(self):
        self.batch += 1
        if self.batch > self.nbatches:
            raise StopIteration()
        return self.data_sampler
    
    # 支持len函数调用对象
    def __len__(self):
        return self.nbatches

class TrainDataLoader(object):
    """
    in_path：路径
    tri_file：三元组文件
    ent_file：实体id文件
    rel_file：关系id文件
    batch_size：一组训练数据的大小
    nbatches：多少个一组
    threads：用多少个线程处理数据
    sampling_mode：抽样本的模式
    bern_flag：是否采用bern的方式构造负样本
    filter_flag：是否过滤掉损坏的三元组
    neg_ent：每个正样本对应的负实体数量，默认为1
    neg_rel：每个正样本对应的负关系数量，默认为0
    """
    def __init__(self, 
		         in_path = "./",
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
		         neg_rel = 0):
        # 加载函数库
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        """argtypes"""
        self.lib.sampling.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64
        ]
        # 设置参数
        self.in_path = in_path
        self.tri_file = tri_file
        self.ent_file = ent_file
        self.rel_file = rel_file
        # 确认文件路径
        if in_path != None:
            self.tri_file = in_path + "train2id.txt"
            self.ent_file = in_path + "entity2id.txt"
            self.rel_file = in_path + "relation2id.txt"
        """set essential parameters"""
        self.work_threads = threads
        self.nbatches = nbatches
        self.batch_size = batch_size
        self.bern = bern_flag
        self.filter = filter_flag
        self.negative_ent = neg_ent
        self.negative_rel = neg_rel
        self.sampling_mode = sampling_mode
        # 交叉采样的flag
        self.cross_sampling_flag = 0
        self.read()

    def read(self):
        # 路劲不为空，采用当前路径。否则采用训练时定义好的路径
        if self.in_path != None:
            self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
        else:
            self.lib.setTrainPath(ctypes.create_string_buffer(self.tri_file.encode(), len(self.tri_file) * 2))
            self.lib.setEntPath(ctypes.create_string_buffer(self.ent_file.encode(), len(self.ent_file) * 2))
            self.lib.setRelPath(ctypes.create_string_buffer(self.rel_file.encode(), len(self.rel_file) * 2))
        
        # 设置属性
        # 是否采用bern进行负样本采样
        self.lib.setBern(self.bern)
        self.lib.setWorkThreads(self.work_threads)
        # 重置随机数生成器
        self.lib.randReset()
        # 导入训练文件
        self.lib.importTrainFiles()
        # 获取关系、实体、三元组的总数
        self.relTotal = self.lib.getRelationTotal()
        self.entTotal = self.lib.getEntityTotal()
        self.tripleTotal = self.lib.getTrainTotal()

        if self.batch_size == None:
            self.batch_size = self.tripleTotal // self.nbatches
        if self.nbatches == None:
            self.nbatches = self.tripleTotal // self.batch_size
        """
        计算了每个批次数据的总大小 self.batch_seq_size。具体来说，它是根据批次大小 self.batch_size、
        负样本数量 self.negative_ent 和负关系数量 self.negative_rel 来计算的。
        注意：这里的批次是指一个正样本产生的测试数据量，在一个批次中，除了包含正样本外，还包含了额外的负样本。
        所以，每个批次中样本的总数量等于正样本数量乘以 (1 + 负样本数量)。
        """
        self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)

        self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)
        self.batch_h_addr = self.batch_h.__array_interface__["data"][0]
        self.batch_t_addr = self.batch_t.__array_interface__["data"][0]
        self.batch_r_addr = self.batch_r.__array_interface__["data"][0]
        self.batch_y_addr = self.batch_y.__array_interface__["data"][0]

    # 普通样本采样
    # 它通过调用 self.lib.sampling 方法来生成样本数据，并返回一个字典，包含生成的批次样本以及模式信息。
    # 具体而言，它会生成一批头部（batch_h）、尾部（batch_t）、关系（batch_r）和标签（batch_y）数据，并将模式设置为 "normal"。
    def sampling(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            0,
            self.filter,
            0,
            0
        )
        return {
            "batch_h": self.batch_h, 
            "batch_t": self.batch_t, 
            "batch_r": self.batch_r, 
            "batch_y": self.batch_y,
            "mode": "normal"
        }

    # 这个函数用于进行头部替换的样本采样
    def sampling_head(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            -1,
            self.filter,
            0,
            0
        )
        return {
            "batch_h": self.batch_h,
            "batch_t": self.batch_t[:self.batch_size],
            "batch_r": self.batch_r[:self.batch_size],
            "batch_y": self.batch_y,
            "mode": "head_batch"
        }

    # 这个函数用于进行尾部替换的样本采样
    def sampling_tail(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            1,
            self.filter,
            0,
            0
        )
        return {
            "batch_h": self.batch_h[:self.batch_size],
            "batch_t": self.batch_t,
            "batch_r": self.batch_r[:self.batch_size],
            "batch_y": self.batch_y,
            "mode": "tail_batch"
        }

    # 头部尾部交叉进行样本生成
    def cross_sampling(self):
        self.cross_sampling_flag = 1 - self.cross_sampling_flag 
        if self.cross_sampling_flag == 0:
            return self.sampling_head()
        else:
            return self.sampling_tail()
            
    """interfaces to set essential parameters"""

    def set_work_threads(self, work_threads):
        self.work_threads = work_threads

    def set_in_path(self, in_path):
        self.in_path = in_path

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.nbatches = self.tripleTotal // self.batch_size

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        self.negative_rel = rate

    def set_bern_flag(self, bern):
        self.bern = bern

    def set_filter_flag(self, filter):
        self.filter = filter

    """interfaces to get essential parameters"""

    def get_batch_size(self):
        return self.batch_size

    def get_ent_tot(self):
        return self.entTotal

    def get_rel_tot(self):
        return self.relTotal

    def get_triple_tot(self):
        return self.tripleTotal

    def __iter__(self):
        if self.sampling_mode == "normal":
            return TrainDataSampler(self.nbatches, self.sampling)
        else:
            return TrainDataSampler(self.nbatches, self.cross_sampling)

    def __len__(self):
        return self.nbatches
    

