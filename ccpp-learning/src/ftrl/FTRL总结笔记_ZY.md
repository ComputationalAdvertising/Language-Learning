## 点击率预估－模型篇

+ author: zhouyongsdzh@foxmail.com
+ date: 20160327

**目录**

+ FTRL-Proximal
+ ADMM + LR 


### FTRL-Proximal

#### FTRL-Proximal由来

+ 在线梯度下降（Online Gradient Descent，OGD）

	可以有非常好的预估准确性，并且占用较少的资源。但是它不能得到有效的稀疏模型（非零参数），就是说添加L1惩罚项也不能产生严格意义上的稀疏解（参数为0）。

+ Truncated Gradient and FOBOS		

	参数\\(w_j\\)设定阈值，当参数小雨阈值时置为0，如此得到稀疏解。
	
+ 正则化对偶平均（Regularized Dual Averaging，RDA）
	
	RDA可以得到稀疏解，在预估准确性和模型稀疏性方面优于FOBOS算法。
		
有没有既能结合RDA获得模型稀疏性，同时又具备OGD的精度的学习算法呢？答案是肯定的，那就是："Follow The (Proximal) Regularized Leader"算法。

#### 模型与参数学习



如果用FTRL-Proximal算法训练LR模型：

+ 首先，预测实例\\(\mathbf{x}^{(i)} \in R^n\\)的label（在给定模型参数\\(\mathbf{w}^{(i)}\\)）。公式为\\(p^{(i)} = \sigma(\mathbf{w}^{(i)} \cdot \mathbf{x}^{(i)}), \; \sigma(a) = \frac{1}{1 + \exp(-a)}\\).

+ 然后，观测标签\\(y^{(i)} \in \{0,1\}\\)，LR模型的损失函数（Logistic Loss）为：

	$$
	\mathcal{l} \; (w^{(i)}) = -y^{(i)} \log p^{(i)} - (1 - y^{(i)}) \log (1- p^{(i)}) \qquad (1)
	$$

	梯度方向：

	$$
	\frac{\partial l \;(w)} {\partial w} = (\sigma(w \cdot x^{(i)}) - y^{(i)}) = (p^{(i)} - y^{(i)}) \cdot \mathbf{x}^{(i)} \qquad(2)
	$$

+ OGD算法迭代公式：

	$$
	w_{i+1} = w_i - \eta_i \mathbf{g}_i \qquad(3)
	$$

	其中\\(\eta_i\\)表示非递增的学习率，如\\(\eta_i=\frac{1}{\sqrt{i}}\\)，\\(\mathbf{g}_i\\)表示当前梯度值。

+ FTRL-Proximal

	$$
	\mathbf{w}_{i+1} = \text{arg} \min_w \left( \underline{ \sum_{s=1}^{i} \mathbf{g}_s} \cdot \mathbf{w} + \frac{1}{2} \sum_{s=1}^{i} \sigma_s {\Vert \mathbf{w} - \mathbf{w}_s \Vert}_2^2 + \lambda_1 {\Vert \mathbf{w} \Vert}_1 \right) \qquad(4)
	$$
	
	\\(\sigma_s\\)为学习率参数，有\\(\sigma_{1:i} = \frac{1}{\eta_i}\\)。上式等价于：
	
	$$
	\left( \sum_{s=1}^{i} \mathbf{g}_s  - \sum_{s=1}^{i} \sigma_s \mathbf{w}_s \right) \cdot \mathbf{w} + \frac{1}{\eta_i} {\Vert \mathbf{w} \Vert}_2^2 + \lambda_1 {\Vert \mathbf{w} \Vert}_1	+ (\text{const})  \qquad(5)
	$$
	
	如果我们存储\\(\mathbf{z}_{i-1} = \sum_{s=1}^{i-1} \mathbf{g}_s  - \sum_{s=1}^{i-1} \sigma_s \mathbf{w}_s\\)，在第\\(i\\)轮迭代时，\\(\mathbf{z}_i = \mathbf{z}_{i-1} + \mathbf{g}_i + \left(\frac{1}{\eta_i} - \frac{1}{\eta_{i-1}} \right) \mathbf{w}_i\\)。求\\(\mathbf{w}_{i+1}\\)每一维参数的闭式解为：
	
	$$
	w_{i+1, j} = 
	\left \{
	\begin{array}{ll}
	0, & \text{if} \; |z_{i,j}| \le \lambda_1 \\
	-\eta_i \left(z_{i,j} - \text{sgn}(z_{i,j}) \lambda_1 \right) & \text{otherwise}.
	\end{array}
	\right. 	\qquad(6)
	$$
	
	每一维特征的学习率计算公式（与对应特征维度梯度累加和 && 梯度平方和相关）：
	
	$$
	\eta_{i,j} = \frac{\alpha}{\beta + \sqrt{\sum_{s=1}^{i} g_{s,j}^2}} = \frac{1}{\sqrt{n_{i+1,j}}}
	\qquad(7)
	$$
	
	\\(n_i = \sum_{s=1}^{i-1} g_{s}^2\\)表示梯度平方累加和。
	
	FTRL-Proximal存储参数\\(\mathbf{z} \in R^n\\)在内存。
	
+ FTRL-Proximal 伪代码

	\\(\{ \\
	\quad \text{Per-Coordinate FTRL-Proximal with L}_1 \text{ and L}_2 \text{ Regularization for Logistic Regression. }  \\
	\quad \text{// 支持} L_1 \text{与} L_2  \text{正则项的FTRL-Proximal算法. Per-Coordinate学习率为公式(7). } \\
	\quad \text{Input: parameters } \alpha, \beta, \lambda_1, \lambda_2. \qquad //  参数 \alpha, \beta 用于\text{Per-Coordinate}计算学习率 \\
	\quad (\text{对于任意的}\; j \in \{1, ..., d\}), 初始化z_i = 0 和n_i=0. \qquad // 总共d个值. \\
	\quad \mathbf{\text{for}} \; t=1 \; to \; T; do \\
	\quad \qquad 接受特征向量\mathbf{x}_t \; and 让 I = \{i| x_i \neq 0 \}. \quad //取非0特征index集合. \\
	\quad \qquad \text{For i} \in I, 计算  \\
	\quad \qquad\qquad w_{t, i} = 
	\quad \left \{
	\quad \begin{array}{ll}
	0, & \text{if} \; |z_{i,j}| \le \lambda_1 \\
	-\left( \frac{\beta + \sqrt{n_i}}{\alpha} + \lambda_2 \right)^{-1} \left(z_{i} - \text{sgn}(z_{i}) \lambda_1 \right) & \text{otherwise}.
	\end{array}
	\right. \\
	\quad \qquad 预测 p_t = \sigma(\mathbf{x}_t \cdot \mathbf{w}) 使用w_{t,i}计算，Predict函数。\\
	\quad \qquad 观测样本 y_t \in \{0, 1\} \qquad //下面更新参数，\text{Update(p, y, x)}函数 \\
	\quad \qquad \text{for all } \; i \in I; do \\
	\quad \qquad\qquad g_i = (p_t - y_t) x_i	\quad \qquad // 第i维特征的梯度值（libsvm格式数据，x_i多为1.）\\
	\quad \qquad\qquad \sigma_i = \frac{1}{\alpha} \left(\sqrt{n_i + g_i^2} - \sqrt{n_i} \right) \qquad // n_i表示第i维梯度（前t-1次）的平方累加和 \\
	\quad \qquad\qquad z_i \leftarrow z_{i-1} + g_i - \sigma_i w_{t,i} \qquad // 更新\text{FTRL}目标函数的梯度公式  \\
	\quad \qquad\qquad n_i \leftarrow n_i + g_i^2  \qquad // 更新第i维梯度平方累加和值。\\
	\}
	\\)


	
	FTRL算法添加了per-coordinate学习率，并且支持L2正则项。
	
	+ per-coordinate

		意思是说FTRL学习过程是对参数向量\\(\mathbf{w}\\)每一维分开训练更新的，每一维使用不同的学习率。与\\(n\\)个特征维度使用统一的学习率相比，此种方法考虑到了训练样本本身在不同特征维度上分布不均匀的特点。
		
		> 如果包含\\(w\\)某一个维度特征的训练样本很少，每一个样本都很珍贵，那么该特征维度对应的训练速率可以独自保持比较大的值，每来一个包含该特征的样本，就可以在该样本的梯度上前进一大步，而不需要与其他特征维度的前进步调强行保持一致。
		

#### FTRL tricks

+ 训练数据采样

	点击预估中正负样本比值悬殊很大，通过subsampling可以大大减小训练数据集的大小。
	
	subsampling策略：正样本全部采，负样本使用一个比例\\(r\\)采样。
	
	如果直接使用采样后的数据训练模型，预测时会导致比较大的bias. 解决方案：
	
	模型训练时，对负样本乘以一个权重，即权重直接乘到损失项上面（loss），这样算梯度时，也就带着权重项。
	
	$$
	w_i = 
	\left \{
	\begin{array}{ll}
	1 & \text{event i is in clicked query} \\
	\frac{1}{r} & \text{event i is in a query with no clicks.}
	\end{array}
	\right.
	$$
	
	**先采样减少负样本数目，在训练的时候再用权重弥补负样本，非常不错的想法。**