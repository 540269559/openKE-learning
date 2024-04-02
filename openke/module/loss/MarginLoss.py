import torch
import torch.nn as nn
import torch.nn.functional as F
from .Loss import Loss

class MarginLoss(Loss):

    # adv_temperature:用于计算负样本权重的温度参数
    # margin:边缘值，默认为6.0
	def __init__(self, adv_temperature = None, margin = 6.0):
		super(MarginLoss, self).__init__()
        # 创造张量margin,且不可训练
		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False
        # self.adv_flag 表示能否使用self.adv_temperature
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False
	
    # 用于计算负样本的权重。
    # n_score是负样本的分数
	def get_weights(self, n_score):
		return F.softmax(-n_score * self.adv_temperature, dim = -1).detach()

    # 前向传播函数,用于计算损失
    # p_score:正样本的分数
    # n_score:负样本的分数
	def forward(self, p_score, n_score):
        # 这里表示适用了权重参数
        # 权重值w = softmax(-n × adv)
		if self.adv_flag:
            # 损失值=mean(sum(w x max(0, p - n + margin))) + margin
			return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin
		else:
            # 这里就是max(p-n+margin, 0)
            # 用了个mean表示对所有的损失值求平均,采用平均值会更好的评估模型的性能
			return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin
			
	# 预测函数:用于计算预测分数
	def predict(self, p_score, n_score):
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()