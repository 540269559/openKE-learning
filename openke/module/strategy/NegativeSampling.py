from .Strategy import Strategy

# 这个 NegativeSampling 类是用于负采样（Negative Sampling）策略的，通常在训练词嵌入模型（如Word2Vec）或推荐系统中使用。
# 负采样是一种优化训练过程的技术，通过随机选择一些负样本（即不属于正样本的样本），使得模型更好地学习到正样本的特征。
class NegativeSampling(Strategy):

	def __init__(self, model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0):
		super(NegativeSampling, self).__init__()
		self.model = model
		self.loss = loss
		self.batch_size = batch_size
        # 正则化项的系数
		self.regul_rate = regul_rate
        # L3正则化系数
		self.l3_regul_rate = l3_regul_rate
    
    # 从总得分中获取正样本的得分。它接收模型输出的总得分，并返回正样本的得分
	def _get_positive_score(self, score):
		positive_score = score[:self.batch_size]
		positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
		return positive_score

    # 从总得分中获取负样本的得分。它接收模型输出的总得分，并返回负样本的得分。
	def _get_negative_score(self, score):
		negative_score = score[self.batch_size:]
		negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
		return negative_score

    # 前向传播方法，用于执行一次前向传播计算损失。它接收一个数据批次 data，通过模型计算得分，并调用损失函数计算正样本和负样本的损失。
    # 然后，根据是否设置了正则化项和L3正则化项，分别对损失进行正则化。最后返回计算得到的损失值。
	def forward(self, data):
		score = self.model(data)
		p_score = self._get_positive_score(score)
		n_score = self._get_negative_score(score)
		loss_res = self.loss(p_score, n_score)
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		return loss_res