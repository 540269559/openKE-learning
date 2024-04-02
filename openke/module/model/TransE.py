import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class TransE(Model):

	def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
		super(TransE, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False

    # 用于计算损失函数中的得分。它接收三元组的头实体、尾实体和关系的嵌入向量以及模式（mode），并根据模式计算得分。
    # 在计算得分之前，如果设置了norm_flag为True，则会对实体和关系的嵌入进行归一化处理。
    # 最后，根据模式计算头部和尾部的得分，并返回。
	def _calc(self, h, t, r, mode):
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		if mode != 'normal':
            # 对头实体的嵌入向量 h 进行形状变换。其中，-1 表示根据张量的总元素数量自动确定该维度的大小，
            # r.shape[0] 表示关系的数量，h.shape[-1] 表示嵌入的维度。
            # 这样的形状变换使得头实体的嵌入向量能够在关系维度上进行广播，以便后续的计算。
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
        # 计算范数,并变成一维向量
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

    # 前向传播
	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		score = self._calc(h ,t, r, mode)
		if self.margin_flag:
			return self.margin - score
		else:
			return score

    # 用于计算模型的正则化项。它接收一个数据字典 data，包含了批次的头实体、尾实体和关系信息。
    # 然后，根据头实体、尾实体和关系的嵌入计算L2正则化项的值，并返回。
	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul

    # 预测得分,然后不断收敛
	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()