# coding:utf-8
import os
import ctypes
import numpy as np

class TestDataSampler(object):

	def __init__(self, data_total, data_sampler):
		self.data_total = data_total
		self.data_sampler = data_sampler
		self.total = 0

	def __iter__(self):
		return self

	def __next__(self):
		self.total += 1 
		if self.total > self.data_total:
			raise StopIteration()
		return self.data_sampler()

	def __len__(self):
		return self.data_total

class TestDataLoader(object):

    def __init__(self, in_path = "./", sampling_mode = 'link', type_constrain = True):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        """for link prediction"""
        self.lib.getHeadBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.getTailBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        """for triple classification"""
        self.lib.getTestBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        """set essential parameters"""
        self.in_path = in_path
        self.sampling_mode = sampling_mode
        self.type_constrain = type_constrain
        self.read()


    # 这个函数的主要目的是准备测试数据，包括设置路径、导入数据文件、获取统计信息和初始化数据数组，为后续的测试任务做好准备。
    def read(self):
        # 使用 ctypes 库的 create_string_buffer 方法创建一个 C 风格的字符串缓冲区，
        # 将 Python 中的路径字符串 self.in_path 转换为 C 风格的字符串，并将其传递给 C++ 函数 setInPath()。
        self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
        # 重置随机数生成器
        self.lib.randReset()
        # 导入测试文件
        self.lib.importTestFiles()

        # 这个操作的目的是在存在类型约束时导入类型文件，用于后续的处理
        if self.type_constrain:
            self.lib.importTypeFiles()

        # 分别获取关系总数、实体总数和测试数据总数
        self.relTotal = self.lib.getRelationTotal()
        self.entTotal = self.lib.getEntityTotal()
        self.testTotal = self.lib.getTestTotal()

        # 初始化测试数据数组
        self.test_h = np.zeros(self.entTotal, dtype=np.int64)
        self.test_t = np.zeros(self.entTotal, dtype=np.int64)
        self.test_r = np.zeros(self.entTotal, dtype=np.int64)
        self.test_h_addr = self.test_h.__array_interface__["data"][0]
        self.test_t_addr = self.test_t.__array_interface__["data"][0]
        self.test_r_addr = self.test_r.__array_interface__["data"][0]

        self.test_pos_h = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_t = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_r = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_h_addr = self.test_pos_h.__array_interface__["data"][0]
        self.test_pos_t_addr = self.test_pos_t.__array_interface__["data"][0]
        self.test_pos_r_addr = self.test_pos_r.__array_interface__["data"][0]
        self.test_neg_h = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_t = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_r = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_h_addr = self.test_neg_h.__array_interface__["data"][0]
        self.test_neg_t_addr = self.test_neg_t.__array_interface__["data"][0]
        self.test_neg_r_addr = self.test_neg_r.__array_interface__["data"][0]


    # 样本预测——link_predict
    # 这个函数主要用于在链接预测任务中生成样本。在链接预测任务中，我们通常需要预测缺失的实体或关系。
    def sampling_lp(self):
        res = []
        self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
        res.append({
            "batch_h": self.test_h.copy(), 
            "batch_t": self.test_t[:1].copy(), 
            "batch_r": self.test_r[:1].copy(),
            "mode": "head_batch"
        })
        self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
        res.append({
            "batch_h": self.test_h[:1].copy(),
            "batch_t": self.test_t.copy(),
            "batch_r": self.test_r[:1].copy(),
            "mode": "tail_batch"
        })
        return res

    # 这个函数主要用于在三元组分类任务中生成样本。在三元组分类任务中，
    # 我们需要对给定的三元组进行分类，判断它是否属于正样本还是负样本。
    def sampling_tc(self):
        self.lib.getTestBatch(
            self.test_pos_h_addr,
            self.test_pos_t_addr,
            self.test_pos_r_addr,
            self.test_neg_h_addr,
            self.test_neg_t_addr,
            self.test_neg_r_addr,
        )
        return [ 
            {
                'batch_h': self.test_pos_h,
                'batch_t': self.test_pos_t,
                'batch_r': self.test_pos_r ,
                "mode": "normal"
            }, 
            {
                'batch_h': self.test_neg_h,
                'batch_t': self.test_neg_t,
                'batch_r': self.test_neg_r,
                "mode": "normal"
            }
        ]

    """interfaces to get essential parameters"""

    def get_ent_tot(self):
        return self.entTotal

    def get_rel_tot(self):
        return self.relTotal

    def get_triple_tot(self):
        return self.testTotal

    def set_sampling_mode(self, sampling_mode):
        self.sampling_mode = sampling_mode

    def __len__(self):
        return self.testTotal

    def __iter__(self):
        if self.sampling_mode == "link":
            self.lib.initTest()
            return TestDataSampler(self.testTotal, self.sampling_lp)
        else:
            self.lib.initTest()
            return TestDataSampler(1, self.sampling_tc)
