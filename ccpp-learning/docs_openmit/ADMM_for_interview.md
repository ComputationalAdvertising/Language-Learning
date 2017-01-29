## ADMM for interview

+ author: zhouyongsdzh@foxmail.com
+ date: 20160404

<br>
### 4. ADMM算法框架
---

<br>
#### 4.1. ADMM简介
---

对于一个有约束条件的优化问题来说，通常的求解方式是构造拉格朗日函数，然后利用**交替优化**的思路，不断地迭代参数和对偶变量（乘子）参数updated表达式，使得两者同时达到optimal。这就是对偶上升法（Dual Ascent）求解的核心思想。

对偶上升法有一个优点就是**如果目标函数是可分的，它可以在分布式环境下求解**。但是它也有一个很大的缺点就是**目标函数必须是强凸函数**。这个对于一般的凸函数，如果用对偶上升法求解参数，并不能得到最优解。

为了解决对偶上升法的缺点，在原有的拉格朗日函数基础上加上**惩罚函数项**（二次项），目标就是为了解决Dual Ascent方法对目标函数要求过于严格的问题。这种优化方法称为**增广拉格朗日乘子法（Argument Lagrange Multipliers）**。

>通过引入惩罚函数项（二次项），放松对于\\(f(x)\\)严格凸的假设和其他一些条件，同时使得算法更加稳健。

虽然增广拉格朗日法放松了对目标函数的限制，但也破坏了dual ascent方法通过**分解参数来并行计算**的优势。因为二次项计算时写成矩阵形式，无法用之前的分块形式。

> **当\\(f\\)是separable时，对于Augmented Lagrangians却是not separable的（因为平方项写成矩阵形式无法用之前那种分块形式）**

为了利用**dual ascent带来的目标函数可分解与增广拉格朗日法带来的更一般化的参数求解**性质。人们提出了ADMM算法。ADMM的思想就是说把原变量、目标函数进行拆分，拆分成不同的变量。形式如下：
		
$$
\begin{align}
& min \quad f(x) + g(z)  \\
& s.t \quad Ax + B z = C
\end{align}  \qquad (2.1)
$$

从上面形式确实可以看出，ADMM的思想就是想把primal变量、目标函数拆分，但是不再像dual ascent方法那样，将拆分开的\\(x_i\\)都看做是\\(\mathbf{x}\\)的一部分，后面融合的时候还需要融合在一起。而是最先开始就将拆开的变量分别看做是不同的变量\\(x\\)和\\(z\\)，约束条件也如此处理。这样的好处就是后面不需要一起融合\\(x\\)和\\(z\\)，保证了前面优化过程的可分解性。

ADMM（Alternating Direction Method of Multipliers，交替方向乘子法）算法框架 之所以能在分布式环境用于大规模学习任务，概括起来主要有3点：

+ **目标函数可分**：可用于多个子任务学习

	将一个整体的任务划分为多个子任务进行学习，每个子任务在一个计算节点（worker）上计算，多个worker节点与一个master节点通信。
	
	> worker更新局部参数和对偶变量；master更新全局参数。
	
+ **目标函数结构，引入交叉方向**

	拆分目标函数和原变量，引入交叉方向变量用于局部参数与全局参数的交替优化

+ **约束条件**：用于保证ADMM形式的优化问题与原优化问题等价。

#### 4.2. ADMM + LR 表达式

机器学习模型的优化的目标函数很多是**“损失函数项＋正则项”**的形式，如LR模型，符合ADMM算法框架表达。

LR模型损失函数（Log损失，交叉熵损失）：

   $$
	\begin{align}
	l \,(y^{(i)}, h_w(x^{(i)})) & = - \log \, L(w) \\\
	& = - \sum_{i=1}^{m} \left( y^{(i)} \cdot \log (h_{w}(x^{(i)})) + (1-y^{(i)}) \cdot \log(1- h_{w}(x^{(i)})) \right)
	\end{align} \qquad (1)
	$$
	
改写为ADMM形式（加上\\(\text{L}_1正则化项\\)）：

$$
\begin{align}
& \min_{\theta, w_1, \cdots, w_T} \quad \sum_{t=1}^{T} \frac{1}{m_t} \sum_{i=1}^{m_t} l \,(y^{(i)}, h_{w_t}(x^{(i)})) + \lambda {\Vert \theta \Vert}_1 \\
& \quad s.t.	\qquad w_t = \theta \;(t=1,\cdots,T)
\end{align} 	\qquad(2)
$$

> ADMM算法待优化的目标函数结构形式为：\\(f(x) + g(x)\\)。目的是区分参数为局部参数和全局参数。局部参数在多个节点进行计算，通过全局参数实现数据信息共享（全局参数信息在多机器之间通过allreduce/broadcast通信来体现）。
> 
> 约束条件\\(w_t = \theta\\) 保证了ADMM形式的目标函数与LR损失函数等价。

增广拉格朗日函数（拉格朗日乘子项 + 损失函数项）为：

$$
\sum_{t=1}^{T} \frac{1}{m_t} \, l \left(\mathbf{Y}_t, h_{w_t}(\mathbf{X}_t)  \right) + \lambda {\Vert \theta \Vert}_1 + \sum_{t=1}^{T} \alpha_{t}^T (w_t - \theta) + \frac{\rho}{2} \sum_{t=1}^{T} {\Vert w_t - \theta \Vert}_2^2 \qquad(3)
$$

迭代步骤为：

+ \\(w\\)更新（worker节点）：

$$
\begin{align}
w_t & \longleftarrow \text{arg} \min_w \frac{1}{m_t} \, l \left(\mathbf{Y}_t, h_{w_t}(\mathbf{X}_t) \right) + \alpha_t^T w_t + \frac{\rho}{2} {\Vert w_t - \theta \Vert}_2^2 \\
\Longrightarrow  w_t & \longleftarrow \text{arg} \min_w \frac{1}{m_t} \, l \left(\mathbf{Y}_t, h_{w_t}(\mathbf{X}_t) \right) + \frac{\rho}{2} {\left \Vert w_t - \theta + \frac{1}{\rho} \alpha_t \right \Vert}_2^2
\end{align}
$$

+ \\(\alpha\\)更新（worker节点）

$$
\alpha_t \longleftarrow \alpha_t + \rho(w_t - \theta)
$$


+ \\(\theta\\)更新（master节点）

	$$
	\begin{align}
	\theta &\longleftarrow \arg \min_{\theta} \lambda {\Vert \theta \Vert}_1 -\sum_{t=1}^{T} \alpha^T \theta + \frac{\rho}{2} \sum_{t=1}^{T} {\Vert w_t -\theta \Vert}_2^2 \\
	\Longrightarrow \theta &\longleftarrow 
	\begin{cases}
		\frac{1}{T} \left(\sum_t (w_t + \frac{\alpha_t}{\rho}) - \frac{\lambda}{\rho} \right), & \quad \text{if } \sum_t (w_t + \frac{\alpha_t}{\rho}) \gt \frac{\lambda}{\rho} \\
		\frac{1}{T} \left(\sum_t (w_t + \frac{\alpha_t}{\rho}) + \frac{\lambda}{\rho} \right), & \quad \text{if } \sum_t (w_t + \frac{\alpha_t}{\rho}) \lt -\frac{\lambda}{\rho} \\
		\quad 0, &\quad \text{if  otherwise} .
	\end{cases}
	\end{align}
	$$

	> 软阈值 soft-thresholding.

#### 4.3. 局部参数学习算法－FTRL-Proximal

##### FTRL-Proximal由来

+ 在线梯度下降（Online Gradient Descent，OGD）

	可以有非常好的预估准确性，并且占用较少的资源。但是它不能得到有效的稀疏模型（非零参数），就是说添加L1惩罚项也不能产生严格意义上的稀疏解（参数为0）。

+ Truncated Gradient and FOBOS		

	参数\\(w_j\\)设定阈值，当参数小雨阈值时置为0，如此得到稀疏解。
	
+ 正则化对偶平均（Regularized Dual Averaging，RDA）
	
	RDA可以得到稀疏解，在预估准确性和模型稀疏性方面优于FOBOS算法。
		
有没有既能结合RDA获得模型稀疏性，同时又具备OGD的精度的学习算法呢？答案是肯定的，那就是："Follow The (Proximal) Regularized Leader"算法。

##### 模型与参数学习

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
		

##### FTRL tricks

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

#### 4.4. DLMC-Core与Rabit基础库功能

Rabit是一个用于机器之间通信操作的基础库，是实现了MPI程序中的Allreduce和broadcast操作，并提供了可容错的功能。在ADMM分布式学习过程中，主要也是利用Allreduce和Broadcast操作，用于局部参数累加、将updated全局参数 广播至通信域中的各个worker节点。

```
rabit::op::Max;
rabit::op::Min;
rabit::op::Sum;
rabit::op::BitOR;
```
在MPI中，通信子（communicator）指的是一组可以互相发送消息的进程集合。MPI_Init的其中一个目的，是在用户启动程序时，定义由用户启动的所有进程所组成的通信子。这个通信子称为```MPI_COMM_WORLD```。

DMLC-Core库主要提供了分布式计算的基础功能，比如序列化接口（IO），定义数据块格式，logging和线程安全的功能。

#### 4.5. 调参经验

ADMM大规模机器学习的调参主要是从3个方面：

1. 特征方面

	LR模型用得到的特征还是“人工特征工程”的做法，通过配置脚本进行特征组合，特征ID化等、特征hash。特征hash的维度需要控制，根据**碰撞率 < 千分之一**的原则，选择dim=300w+1.
	
2. 数据方面

	训练数据在分布式环境下的节点加载，一个广告位的预估模型训练只在一个节点上进行，有些广告位数据量很大，内存放不下，这个时间需要对训练进行分块，分批读入。需要确定数据切块的个数。
	
3. 模型参数和训练

	训练过程的迭代次数，FTRL中正则化项系数\\(\lambda_1, \lambda_2, \rho\\)等。
	
	迭代次数根据训练时间和收敛情况（auc和logloss指标），选取10. 训练时间：1h左右。
	
	参数根据是否overfitting情况，来调整。结论：\\(\lambda_2 = 200, \lambda_1 = 1, \rho=1 \text{ (step size)}\\)
	
#### 4.6. 指标

+ offline：auc + logloss
+ online：流量切分abtest实验，线上平均点击率，通过dashboard观察；

#### 4.7. ADMM模型线上计算

线上计算过程：模型解析成字典树，对应叶子节点就是特征值的权重。线上计算ctr的过程，就转化为根据当前cookie对应的特征取值遍历字典树的过程。将得到的叶子节点累加，就得到了ctr结果。

+ **字典树**：离线模型从redis中读取，线上加载并组织成字典树

	字典树，又称单词查找树，是一种hash树的变种。 字典树保存一些字符串->值的对应关系。他的最大优点是：**利用字符串的公共前缀来节约存储空间，能最大限度地减少无谓的字符串比较，查询效果比hash表高。**
	
	 Trie的核心思想是空间换时间。利用字符串的公共前缀来降低查询时间的开销以达到提高效率的目的。
	 
	 ```
	 {
    "list": [
      {
        "fid": "0", 
        "key": "P_N_T_1_20631_5d3eb4db5df43584a3c5cb5d3192493c", 
        "value": -0.0051592299999999999
      }, 
      {
        "fid": "0", 
        "key": "P_N_T_1_20323_a0971a5a658b38489a90e9a28434bcdf", 
        "value": 0.0065683900000000003
      }, 
      {
        "fid": "0", 
        "key": "P_N_T_1_20426_c14e4136cc5a313f8589c9fa339dc1b8", 
        "value": 0.054301799999999997
      }, 
    "type": "ps_cmi_ad"
}
```

+ 线上模型的计算流程

	+ redis查询，得到候选商品信息＋用户属性信息
	+ 根据特征遍历字典树模型，得到的叶子节点（对应特征value）值进行累加，得到的值为ctr值；

	这样会得到k个商品对应的ctr值：
	
	```
	{
		"cookie":"cookie1",
		"ctr": [
			"uuid_1":"ctr_1",
			"uuid_2":"ctr_2",
			...,
			"uuid_k":"ctr_k"
		]
	}
	```
	+ 按照ctr值（或结合其它因素）排序，返回topK商品。	
	
	