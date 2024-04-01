# coding:utf-8
import torch
from torch.autograd import Variable
import torch.optim as optim
import os
from tqdm import tqdm

class Trainer(object):
    """
    参数解析:
    model:模型对象
    data_loader:使用的loader对象
    train_times:训练次数
    alpha:margin的值
    use_gpu:是否使用GPU
    opt_method:优化方法(SGD:随机梯度下降)
    save_steps:多少次保存一次临时结果
    checkpoint_dir:临时存储的文件位置
    """
    def __init__(self, 
		         model = None,
		         data_loader = None,
		         train_times = 1000,
		         alpha = 0.5,
		         use_gpu = True,
		         opt_method = "sgd",
		         save_steps = None,
		         checkpoint_dir = None):
        
        # 多线程
        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        # 优化器(对象)
        # 会使用到optim库提供的优化方法
        # 例如:optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
        # optimizer = optim.Adam([var1, var2], lr = 0.0001)
        self.optimizer = None
        # 学习率衰减,主要也是应用到optimizer上的
        self.lr_decay = 0
        # 权重衰减,主要也是应用到optimizer上的
        self.weight_decay = 0
        self.alpha = alpha

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir
    
    # 训练函数
    def run(self):
        # 是否采用GPU跑
        if self.use_gpu:
            self.model.cuda()
        
        # 如果已经有了优化器，那就直接用优化器，不然就要参考优化的方案，构造优化器
        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay
            )

        print("Finish initializing...")
        # 这里主要是用于可视化界面的展示，有助于使用者观察到训练的程度
        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            res = 0.0
            for data in self.data_loader:
                # 计算loss值
                loss = self.train_one_step(data)
                res += loss
            training_range.set_description("Epoch %d | loss: %f" % (epoch, res))

            # 是否需要存储临时的结果
            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))



    # 训练过程中的一个步骤
    def train_one_step(self, data):
        # 1.梯度清零,避免梯度积累
        self.optimizer.zero_grad()
        # 2.将数据展示成模型相应的参数并传入给模型
        # batch_h：h向量
        # batch_t：t向量
        # batch_r：r向量
        # mode：处理数据的模式
        # 这里应该是调用self.mode.forward方法
        loss = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
			'batch_t': self.to_var(data['batch_t'], self.use_gpu),
			'batch_r': self.to_var(data['batch_r'], self.use_gpu),
			'batch_y': self.to_var(data['batch_y'], self.use_gpu),
			'mode': data['mode']
        })
        # 3.反向传播
        # 对损失值进行反向传播，计算出模型参数的梯度
        loss.backward()
        # 4.参数更新
        # 根据优化器的规则更新模型参数，以最小化损失
        self.optimizer.step()
        return loss.item()

    """
    函数作用:
        将 NumPy 数组 x 转换为 PyTorch 的 Tensor 对象，并将其封装在一个 Variable 对象中
        如果use_gpu为True,就会将这个计算移动到GPU中加速
    参数解析:
    x为numpy数组
    use_gpu表示是否采用GPU
    """
    def to_var(self, x, use_gpu):
        # 这里转成tensor然后存入Variable变量会更方便进行后续操作
        # 比如；直接对Variable进行反向传播
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    # 更改私有属性的函数
    def set_model(self, model):
        self.model = model
    
    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
    
    def set_alpha(self, alpha):
        self.alpha = alpha
    
    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir = None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

