## 参数学习算法

+ author: zhouyongsdzh@foxmail.com
+ date: 20151027

**目录**

+ [0. 写在前面](#0.写在前面)
+ [1. 最优化问题求解](#1.最优化问题求解)
    + [1.1. Gradient Descent](#1.1.Gradient Descent)
    + [1.2. Adaptive Gradient Descent](#1.2.Adaptive Gradient Descent) 
    + [1.3. AdaDelta](#1.3.AdaDelta)
    + [1.4. Follow The Regularized Leader-Proximal](#1.4.Follow The Regularized Leader)
+ [2. 概率推理](#2.概率推理)
    + MCMC  

<h2 id="0.写在前面">0. 写在前面</h2>
 
<h2 id="1.最优化问题求解">1. 最优化问题求解</h2> 

<h3 id="1.4.Follow The Regularized Leader">1.4. Follow The Regularized Leader-Proximal</h3>

FTRL-Proximal算法是Google在2013年的一篇论文《Ad Click Prediction: a View from the Trenches》中提到的参数在线学习算法，论文中提到该算法用在了Google的搜索广告在线学习系统中。因为算法比较容易理解且工程实现不复杂，业内诸多公司都有尝试并取得了不错的收益。

FTRL-Proximal算法不仅可用于在线学习（不要求数据IID，样本序列学习），同时也可以用于离线基于batch数据的参数求解。所以这里也把该算法作为“优化器”的一种，用于大规模机器学习任务的参数求解（该算法本身就是为了很好的求解大规模在线学习任务而提出的）。

#### 1.4.1. FTRL-Proximal演化进程

+ **Online Gradient Descent (OGD, 在线梯度下降）**

    可以有非常好的预估准确性，并且占用较少的资源。但是它不能得到有效的稀疏模型（非零参数），即便优化目标添加L1惩罚项也不能产生严格意义上的稀疏解（会使参数接近于0，而不是0）。
    
    > 这里OGD其实就是随机梯度下降，之所以强调Online是因为这里不适用于解决batch数据问题的，而是用于样本序列化问题求解的，后者不要求样本是独立同分布（IID））的。
    
+ **Truncated Gradient and FOBOS**
    参数\\(w_j\\)设定阈值，当参数小雨阈值时置为0，可得稀疏解。
	
+ **Regularized Dual Averaging（RDA, 正则化对偶平均）**
	RDA可以得到稀疏解，在预估准确性和模型稀疏性方面优于FOBOS算法。
		
有没有既能结合RDA获得模型稀疏性，同时又具备OGD的精度的学习算法呢？答案是肯定的，那就是："Follow The (Proximal) Regularized Leader"算法。

#### 1.4.2. FTRL-Proximal工作原理

为了更容易的理解FTRL的工作原理，这里以学习LR模型参数为例，看FTRL是如何求解的：

+ 首先，对于一个实例\\(\mathbf{x}^{(i)} \in R^n\\)预测其标签$y=1$的概率$p$（第$i$次迭代的模型参数\\(\mathbf{w}^{(i)}\\)）。公式\\(p^{(i)} = \sigma(\mathbf{w}^{(i)} \cdot \mathbf{x}^{(i)}), \; \sigma(a) = \frac{1}{1 + \exp(-a)}\\).

+ 然后，观测标签\\(y^{(i)} \in \{0,1\}\\)，LR模型的损失函数（Logistic Loss）为：

	$$
	\mathcal{l} \; (w^{(i)}) = -y^{(i)} \log p^{(i)} - (1 - y^{(i)}) \log (1- p^{(i)}) \qquad (1)
	$$
	
	梯度方向：

	$$
	\mathbf{g}_i = \nabla_i = \frac{\partial l \;(w)} {\partial w} = (\sigma(w \cdot x^{(i)}) - y^{(i)}) = (p^{(i)} - y^{(i)}) \cdot \mathbf{x}^{(i)} \qquad(2)
	$$

    OGD算法迭代公式：

	$$
	w_{i+1} = w_i - \eta_i \mathbf{g}_i \qquad(3)
	$$

	其中\\(\eta_i\\)表示非递增的学习率，如\\(\eta_i=\frac{1}{\sqrt{i}}\\)，\\(\mathbf{\nabla}_i\\)表示当前梯度值。

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
	0, & \text{if} \; |z_{i,j}| \le \lambda_1 \\\
	-\eta_i \left(z_{i,j} - \text{sgn}(z_{i,j}) \lambda_1 \right) & \text{otherwise}.
	\end{array}
	\right. 	\qquad(6)
	$$
	
	每一维特征的学习率计算公式（与对应特征维度梯度累加和 && 梯度平方和相关）：
	
	$$
	\eta_{i,j} = \frac{\alpha}{\beta + \sqrt{\sum_{s=1}^{i} g_{s,j}^2}} = \frac{1}{\sqrt{n_{i+1,j}}}
	\qquad(7)
	$$
	
	\\(n_i = \sum_{s=1}^{i-1} g_{s}^2\\)表示梯度平方累加和，参数\\(\mathbf{z} \in R^n\\)在内存存储。

#### 1.4.3. FTRL-Proximal伪代码

\\(\{ \\\
    \quad01. \;\text{Per-Coordinate FTRL-Proximal with L}_1 \text{ and L}_2 \text{ Regularization for Logistic Regression. }  \\\
	\quad02. \;\text{// 支持} L_1 \text{与} L_2  \text{正则项的FTRL-Proximal算法. Per-Coordinate学习率为公式(7). } \\\
	\quad03. \;\text{Input: parameters } \alpha, \beta, \lambda_1, \lambda_2. 初始化z_i = 0 和n_i=0.\quad //  参数 \alpha, \beta 用于\text{Per-Coordinate}计算学习率 \\\
	\quad04. \;\mathbf{\text{for}} \; t=1 \; to \; T; do \\\
	\quad05. \quad 接受特征向量\mathbf{x}_t, \; I = \{i| x_i \neq 0 \}. \quad //取非0特征index集合。 \\\
	\quad06. \quad\text{For i} \in I, 计算  \\\
	\quad07. \quad\qquad w_{t, i} = 
	\; \left \{
	\; \begin{array}{ll}
	0, & \text{if} \; |z_{i}| \le \lambda_1 \\\
	-\left( \frac{\beta + \sqrt{n_i}}{\alpha} + \lambda_2 \right)^{-1} \cdot \left(z_{i} - \text{sign}(z_{i}) \lambda_1 \right) & \text{otherwise}
	\end{array}
	\right.  。\\\ 
	\quad08. \quad 预测 p_t = \sigma(\mathbf{x}_t \cdot \mathbf{w}) 使用w_{t,i}计算。\\\
	\quad09. \quad \text{For} \; i \in I; 更新参数 \\\
	\quad10. \qquad g_i = (p_t - y_t) \cdot x_i \qquad // 第i维特征的梯度（libsvm格式数据，x_i多为1.）\\\
	\quad11. \qquad \sigma_i = \frac{1}{\alpha} \left(\sqrt{n_i + g_i^2} - \sqrt{n_i} \right) \qquad // n_i表示第i维梯度（前t-1次）的平方累加和 \\\
	\quad12. \qquad z_i \leftarrow z_{i-1} + g_i - \sigma_i w_{t,i} \qquad // 更新\text{FTRL}目标函数的梯度公式  \\\
	\quad13. \qquad n_i \leftarrow n_i + g_i^2  \qquad // 更新第i维梯度平方累加和值。\\\
	\}
	\\)
	
FTRL算法添加了per-coordinate学习率，目标函数支持L2正则。
	
+ 解释per-coordinate

    FTRL学习过程是对参数向量\\(\mathbf{w}\\)每一维分开训练更新的，每一维使用不同的学习率。与\\(n\\)个特征维度使用统一的学习率相比，此种方法考虑到了训练样本本身在不同特征维度上分布不均匀的特点。
		
    > 如果包含\\(w\\)某一个维度特征的训练样本很少，每一个样本都很珍贵，那么该特征维度对应的训练速率可以独自保持比较大的值，每来一个包含该特征的样本，就可以在该样本的梯度上前进一大步，而不需要与其他特征维度的前进步调强行保持一致。
    
#### 1.4.3. FTRL实验经验

| 参数  | 变化 | $\mathcal{\sigma}$变化 | $z$变化 | $n$变化 | lrate变化 | $w$变化 | 备注 |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | --- |
| $\alpha$ | $\uparrow$ | $\downarrow$ | 振幅变小 | $-$ | $\uparrow$ | 振幅不确定 | 
| $\beta$ | $\uparrow$ | $-$ | $-$ | $-$ | $\downarrow$ | 振幅变大 | 实践$\beta$从0.1调至2，效果正向 |
| $\mathcal{l_1}$ | $\uparrow$ | $-$ | $-$ | $-$ | $-$ | 稀疏性增加，绝对值变小 | 从0.5调到0.01效果正向 |
| $\mathcal{l_2}$ |
| $\nabla_g$ | $\downarrow$ | $\downarrow$ | 振幅变小 | 变小$\downarrow$ | 变大 | 

1. alpha持续调大，训练集拟合越好越好，测试集表现变差，原因分析: 

    > alpha变大，sigma变小，z的振幅变小
    
    
2. 数据

 

非常赞：优化算法总结文档：http://sebastianruder.com/optimizing-gradient-descent/
优化算法：http://www.cnblogs.com/neopenx/p/4768388.html

### [FM＋SGD](http://blog.csdn.net/itplus/article/details/40536025)



