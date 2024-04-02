import torch
import torch.nn as nn
import os
import json

# 这个对象是一个基本的神经网络模型类，包含了加载和保存模型参数的方法，以及获取和设置模型参数的功能。
class BaseModule(nn.Module):

    # 初始化函数
    # 定义0向量和pi的取值
    def __init__(self):
        super(BaseModule, self).__init__()
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        # 表示在反向传播过程中不对其进行梯度更新
        self.zero_const.requires_grad = False
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        self.pi_const.requires_grad = False

    def load_checkpoint(self, path):
        # 从指定路径加载模型的参数
        # 加载模型参数的字典,将参数加载到模型中。
        self.load_state_dict(torch.load(os.path.join(path)))
        # 将模型设置为评估模式
        self.eval()

    # 将模型的参数保存到指定路径。
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    # 从指定路径加载模型参数。
    def load_parameters(self, path):
        f = open(path, "r")
        parameters = json.loads(f.read())
        f.close()
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict = False)
        self.eval()

    # 将模型的参数保存到指定路径。
    def save_parameters(self, path):
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    # 获取模型的参数。
    # 根据参数 mode 的设置，将参数转换为不同的数据类型，支持 numpy 数组和列表格式。
    # 可以指定要获取的参数字典，如果未指定，则获取所有参数。
    def get_parameters(self, mode = "numpy", param_dict = None):
        all_param_dict = self.state_dict()
        if param_dict == None:
            param_dict = all_param_dict.keys()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = all_param_dict[param].cpu().numpy()
            elif mode == "list":
                res[param] = all_param_dict[param].cpu().numpy().tolist()
            else:
                res[param] = all_param_dict[param]
        return res
    # 设置模型的参数。
    def set_parameters(self, parameters):
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict = False)
        self.eval()