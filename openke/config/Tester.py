# coding:utf-8
import torch
from torch.autograd import Variable
import os
import ctypes
from tqdm import tqdm
import numpy as np

class Tester(object):
    def __init__(self, model = None, data_loader = None, use_gpu = True):
        # 设置参数
        self.model = model 
        self.data_loader = data_loader 
        self.use_gpu = use_gpu 

        # 获取基础文件
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        # 将Base.so加载为动态链接库并且放入到self.lib这个属性中
        """
        关于ctypes的使用介绍
        在 Linux 中，要求指定文件名包括扩展名来加载库，因此不能使用属性访问的方式来加载库。
        你应当使用 dll 加载器的 LoadLibrary() 方法，或是应当通过调用构造器创建 CDLL 的实例来加载库:
        """
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        # 这两行设置了testHead和testTail函数的参数类型。它们告诉Python这些函数接受的参数类型是什么。
        # 在这里，它们告诉Python这些函数接受一个指向void的指针，以及两个64位整数作为参数。
        """
        调用可变函数
        在许多平台上通过 ctypes 调用可变函数与调用带有固定数量形参的函数是完全一样的。 
        在某些平台，特别是针对 Apple 平台的 ARM64 上，可变函数的调用约定与常规函数则是不同的。
        在这些平台上要求为常规、非可变函数参数指定 argtypes 属性:
        """
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int16, ctypes.c_int16]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int16, ctypes.c_int16]
        # 这一行设置了test_link_prediction函数的参数类型。它告诉Python这个函数接受一个64位整数作为参数。
        # .argtypes属性用于设置函数的参数类型
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        # 设置各种属性传入的参数类型
        """
        1.MRR(平均倒数排名)
        2.MR(平均排名)
        3.hit@1(正确结果排名前1的概率)
        4.hit@3(正确结果排名前3的概率)
        5.hit@10(正确结果排名前10的概率)
        """
        # ctypes.c_int64代表C64位signed int的数据类型
        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        # 默认情况下都会假定函数返回 C int 类型。 
        # 其他返回类型可通过设置函数对象的 restype 属性来指定。
        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        # 是否使用GPU进行模型训练
        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model): 
        self.model = model

    def data_loader(self, data_loader):
        self.data_loader = data_loader

    def use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()
    
    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))
    
    def test_one_step(self, data):
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
			'batch_t': self.to_var(data['batch_t'], self.use_gpu),
			'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })
    
    # 主要用于测试集的预测，同时产生特定的评估指标返回
    def run_link_prediction(self, type_constrain = False):
        self.lib.initTest()
        # 这一行设置了data_loader对象的采样模式为'link'，
        # 还有另一种模式则是属于三元组分类数据的采集
        # 表明接下来的数据加载将用于链接预测。
        self.data_loader.set_sampling_mode('link')
        # 是否有类型限制
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0

        # 加载条画面
        training_range = tqdm(self.data_loader)
        # 每一组的测试都会更换所有头部和尾部，因此每一个向量都会有很多的向量
        # 这里的data_head就是代表当前向量更换所有(或过滤后)头部的向量
        for index, [data_head, data_tail] in enumerate(training_range):
            score = self.test_one_step(data_head)
            """
            __array_interface__的存在使得NumPy数组可以以一种通用的方式在不同的编程语言和库之间传递和共享数据。
            例如，如果你想要将NumPy数组传递给一个不支持NumPy的库，
            你可以通过__array_interface__提供的信息来构建对应的数据结构，
            从而在不同的环境中共享数据。
            """
            self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            score = self.test_one_step(data_tail)
            self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
        
        self.lib.test_link_prediction(type_constrain)

        mrr = self.getTestLinkMRR(type_constrain)
        mr = self.getTestLinkMR(type_constrain)
        hit10 = self.getTestLinkHit10(type_constrain)
        hit3 = self.getTestLinkHit3(type_constrain)
        hit1 = self.getTestLinkHit1(type_constrain)
        print(hit10)

        return mrr, mr, hit10, hit3, hit1

    # 这个函数是用来获取最佳阈值的，用于二分类任务中
    # 其中，score是预测的分数，ans是真实的样本（0或1）
    def get_best_threshlod(self, score, ans):
        # 将ans和score按列合并，形成一个二维数组，每一行对应一个样本
        # 第一列为真实标签，第二列为预测的分数
        res = np.concatenate([ans.reshape(-1, 1), score.reshape(-1, 1)], axis=-1)
        # 按照score进行排序
        order = np.argsort(score)
        res = res[order]

        # 计算总样本数
        total_all = (float)(len(score))
        total_current = 0.0
        # 计算正例数
        total_true = np.sum(ans)
        # 计算负例数
        total_false = total_all - total_true

        # 初始化了最大分数res_mx
        res_mx = 0.0
        # 初始化对应的阈值
        threshlod = None
        for index, [ans, score] in enumerate(res):
            # 计算正例的个数
            if ans == 1:
                total_current += 1.0
            # 计算当前阈值下的分数
            """
            加上total_false即假正例的数量。在二分类问题中，假正例是我们希望减少的错误，因此在评估阈值时，我们希望考虑到假正例的影响
            目的是为了在计算时考虑到假正例的权重
            减去index+1是为了对错误分类的样本进行一定的惩罚,与2*total_current相对应
            综上:加入total_false是为了考虑假正例的影响,而减去Index+1则是为了对错误分类的样本进行一定的惩罚,以便全面地评估当前阈值的效果
            """
            res_current = (2 * total_current + total_false - index - 1) / total_all
            # 记录最大F1分数和对应的阈值
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    # 进行三元组二分类
    def run_triple_classification(self, threshlod = None):
        # 初始化测试集
        self.lib.initTest()
        # 设置数据加载模式为二分类
        self.data_loader.set_sampling_mode('classification')
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        for index, [pos_ins, neg_ins] in enumerate(training_range):
            # 对正样例进行测试
            res_pos = self.test_one_step(pos_ins)
            # 对正例进行一步测试，并将结果添加到score列表中，同时将对应的真实标签（1）添加到ans列表中。
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)
            # 对负例进行一步测试，并将结果添加到score列表中，同时将对应的真实标签（0）添加到ans列表中。
            # 注意，这里ans的长度是根据正例的数量来确定的，因此对负例也采用了相同的长度。
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        # 转换成numpy数组
        score = np.concatenate(score, axis = -1)
        ans = np.array(ans)

        # 如果没有阈值,则调用方法获取
        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        # 合并排序
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            # 如果当前阈值下的分数大于预设阈值，则计算准确率
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod

