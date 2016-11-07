title: "第5章：DiMLSys-算法框架-ADMM" 
mathjax: true
date: 2015-01-05 18:30:31
categories: 
	- 分布式机器学习
tags: 
	- ADMM
	- 交叉方向乘子法
	- 梯度提升法
	- 受限约束优化
---

+ author: zhouyongsdzh@foxmail.com
+ date: 2016-03-29
+ weibo: [@周永_52diml](http://weibo.com/p/1005051707438033/home?)

---

**内容列表**

+ 写在前面
+ 分布式环境下的机器学习问题
+ 约束优化问题
	+ 对偶提升
	+ 对偶分解
	+ 增强拉格朗日乘子法
+ 交叉方向乘子法
+ 理论证明
+ 一致性与共享（Consensus and Sharing）
+ ADMM与统计学习
	+ 适用于哪些学习问题：统计学习（GLM，Addictive Model） 
+ ADMM与机器通信
	+ ADMM与同步通信机制（以rabit为例）
	+ ADMM与异步通信机制（以ps为例）
+ ADMM应用场景
	+ 分布式机器学习任务（大数据学习）
	+ 多任务学习问题 
+ 参考资料

### 写在前面

分布式机器学习系统，一定要有链接硬件的软组织－分布式算法框架。这里要介绍的分布式算法框架是ADMM算法（Alternating Direction Multiplier Method，交叉方向乘子法）。

ADMM算法结构天然地适用于分布式环境下具体任务的求解。在详细介绍ADMM分布式算法之前，我们先了解狭下一个大学习问题的如何在分布式环境下拆分成多个子任务学习问题的。然后通过《约束优化问题一般的解决方案》来阐述ADMM算法的演化过程，过渡到ADMM。最后详细阐述ADMM算法结构、理论可行性证明、分布式环境下如何保证一致性以及信息共享，适用的学习问题。

**ADMM算法问题：**

1. ADMM算法的定位是什么？
2. ADMM算法为什么适合作为解决分布式机器学习任务的算法框架？ 
3. ADMM前世今生？（对偶提升，对偶分解，增强拉格朗日法）
4. ADMM算法框架的结构表达？
5. ADMM算法在分布式环境下的理论体系？（收敛性，一致性）
6. ADMM算法在分布式环境下工作过程？（以yarn为例）
7. 哪些学习模型可以适用于ADMM算法？
8. AFMM算法使用场景？

### 分布式环境下的机器学习问题

1. 整个的学习任务表达式
2. 任务拆解成多个子任务，通过数据切分

### 约束优化问题一般解决方案

1. 对偶提升法的更新逻辑是什么？参数的物理意义？
2. 为什么对偶提升法对目标函数有较强的约束？
3. Augmented Lagrangian法添加二次项后 为什么可以缓解约束？
4. 为什么Augmented Lagrangian不能分解 ？是参数不能分解还是二次项不能分解？
5. ADMM算法框架为什么可以分解？

约束优化问题可以分为两大类：等式约束优化和不等式约束优化。我们这里以等式约束优化为例，一个典型的等式约束优化问题，形式化表示如下：

$$
\begin{align}
& \; \min_{x} \;\; f(x) \\\
& s.t. \; Ax = b
\end{align} 	\qquad\qquad(diml.2.5.1)
$$

其中，\\(f(x)\\)是待优化的**目标函数**，\\(Ax=b\\)作为优化目标函数时需要满足的**约束条件**。\\(s.t\\)是英文```subject to```的缩写。

通常，求解等式约束优化问题最简单的方法之一－对偶提升法。

<br>
#### 对偶提升法

对偶提升法（Dual Ascent）的核心思想是通过引入一个**对偶变量**，利用**交替优化**的思路，使得两个同时达到最优。

> 对偶变量即是拉格朗日乘子，又称算子。

对公式\\((diml.2.5.1)\\)引入拉格朗日乘子（用\\(\beta\\)表示），得到拉格朗日公式为：

$$
\mathcal{L}(x, \beta) = f(x) + \beta(Ax-b)    \qquad\qquad(diml.2.5.2)
$$

对偶函数用\\(g(\beta)\\)表示： 

$$
g(\beta) = \inf_{x} \mathcal{L}(x, \beta) = -f^{*}(-A^T\beta) - b^T\beta  \qquad\\qquad(diml.2.5.3)
$$

对偶提升就是最大化对偶函数，即\\(\text{maximize}\; g(\beta)\\)。在**强对偶性**假设下，原函数和对偶函数会同时达到最优，可以得到：

$$
x^{*} = \arg \min_{x} \mathcal{L}(x, \beta^{*})   \qquad\qquad\quad(diml.2.5.4)
$$

> 什么是强对偶性？
> 
> 如果最小化**原凸函数**等价于最大化**对偶函数**时，称为强对偶性。即\\(\min f(x) = \max g(\beta)\\)

如果对偶函数\\(g(\beta)\\)可导，使用Dual Ascent法，交替更新参数，使其同时收敛到最优。参数更新的迭代公式如下：

$$
\begin{align}
x^{k+1} & := \arg \min_{x} \mathcal{L}(x, \beta^{k})  \qquad\qquad\;(\text{step1}) \\\
\beta^{k+1} & := \beta^k + \alpha^{k}(A x^{k+1} - b) \qquad\quad(\text{step2})
\end{align} \qquad(diml.2.5.5)
$$

> 公式\\((diml.2.5.5)\\)解释：
> 
> step1: 求拉格朗日函数极小化时对应的参数\\(x\\)，具体实现时，需要对参数\\(x\\)求偏导，令\\(\frac{\partial}{\partial{x}} L(x, \beta^k) = 0\\).
> 
> step2: 公式\\((diml.2.5.3)\\)对\\(\beta\\)求偏导得到梯度，使用梯度提升法得到对偶变量\\(\beta\\)的迭代公式，\\(\alpha^k\\)为迭代步长。这一步称为对偶提升。

对偶提升法在满足强对偶性假设下可以证明公式\\((diml.2.5.5)\\)能达到收敛。即要求目标函数\\(f(x)\\)为强凸函数（一般函数难以满足）。

> 什么是强凸函数？（在[《深入强出机器学习》系列 第10章：深入浅出ML之cluster家族 中的EM算法]()中有提到）
> 
> 强凸函数需满足：\\(E[f(x)] > f(E(x)) \\)
> 
> 函数\\(f: I \rightarrow R\\)成为强凸的，若\\(\exists\alpha > 0\\)，使\\(\forall(x, y) \in I \times I, \forall t \in [0, 1]\\)，恒有：
> 
$$
f[tx+(1-t)y] \le tf(x) + (1-t) f(y) - t(1-t) \alpha (x-y)^2
$$

<br>
#### 对偶分解

Dual Ascent法有一个非常好的性质：

**当目标函数\\(f\\)可分（separable）时，整个优化问题可以拆分成多个子优化问题，分块优化后得到局部分数，然后汇集起来 整体更新全局参数，有利于问题的并行化处理。这个过程称为对偶分解（Dual Decomposition）**

对公式\\((diml.2.5.1)\\)拆分表示为：

$$
\begin{align}
& \min_{x} f(x) = \sum_{i=1}^{k} f_i(x_i) \\\
& s.t. \; Ax = \sum_{i=1}^{k} A_i x_i = b 
\end{align}   \qquad\qquad(diml.2.5.6)
$$

对应的拉格朗日函数：

$$
\mathcal{L}(x, \beta) = \sum_{i=1}^{k} \mathcal{L_i}(x_i, \beta) = \sum_{i=1}^{k} \left(f_i(x_i) + \beta^T A_i x_i - \frac{1}{N}\beta^T b \right)  \qquad(diml.2.5.7)
$$

对应的参数更新公式：

$$
\begin{align}
x_{i}^{k+1} & := \arg \min_{x} L_i(x_i,\beta^{k}) \qquad\qquad\qquad\quad\qquad(step1)\\\
\beta^{k+1} & := \beta^{k} + \alpha^k \nabla g(\beta) = y^k + \alpha^k(A x^{k+1} -b)  \qquad\,(step2)
\end{align} \qquad\qquad(diml.2.5.8)
$$

> step1: 并行化求解多个子目标函数
> 
> step2: 汇总\\(x_{i}^{k+1}\\)（论证是否求平均可否？），更新对偶变量.


<br>
#### 增广拉格朗日乘子法

Dual Ascent方法求解问题时，目标函数必须满足强凸函数的条件，限制过于严格。很多目标函数是不满足强凸函数条件的，为了满足这部分目标函数的极值求解问题，使用[增广拉格朗日乘子法（Augmented Lagrangians）](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method)可以解决。

> 增广拉格朗日乘子法的目标：**解决Dual Ascent方法对目标函数要求过于严格的问题。**

Argumented Lagrangians方法的核心思想：

| 通过引入惩罚函数项（二次项），放松对目标函数\\(f(x)\\)严格凸的限制，同时使得算法更加稳健。|
| --- |

具体做法：在原有的拉格朗日公式中添加惩罚函数项。即：

$$
\mathcal{L}_{\rho}(x, \beta) = f(x) + \beta^T (Ax-b) + \frac{\rho}{2} {\Vert Ax-b \Vert}_2^2     \qquad\quad(diml.2.5.9)
$$

> 公式解读：
> 
$$
\mathcal{L}_{\rho}(x, \beta) = \overbrace{f(x)}^{目标函数} + \overbrace{ \underbrace{\beta^T(Ax-b)}_{拉格朗日乘子项} + \underbrace{\frac{\rho}{2} {\Vert Ax-b \Vert}_2^2}_{惩罚函数项} }^{增广拉格朗日乘子项}
$$

对应的参数迭代公式：

$$
\begin{align}
x^{k+1} & := \arg \min_{x} \mathcal{L}(x, \beta^k) \qquad\qquad\qquad\qquad (step1) \\\
\beta^{k+1} & := \beta^k - \alpha^k \nabla g(\beta) = \beta^k - \rho(Ax^{k+1}-b)  \quad\;(step2)
\end{align}  \qquad(diml.2.5.10)
$$

因为惩罚函数项是一个二次项，step1求解释时 写成矩阵形式，无法用分块形式进行并行化求解。

虽然增广拉格朗日乘子法放松了对目标函数的限制，但是也破坏了Dual Ascent方法利用分解参数来并行求解的优化。即便当目标函数\\(f(x)\\)是separable时，对于Augmented Lagrangians却是not separable的。

如果能结合Dual Ascent的并行求解的优势 和 Augmented Lagrangians优秀的收敛性质，那么就可以使得大多数目标函数求解都能在分布式环境下并行实现了，岂不美哉！！！

果然在2010年由Stephen Boyd大师等人提升了ADMM算法，可以解决上述的问题。为此它们长篇论述了ADMM算法的演化过程，参考论文：[Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers](http://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf) 该文被《Foundations and Trends in Machine Learning》录入。

> 从文章题目上可以看出ADMM算法的适用场景：分布式优化 && 统计学习。
> 
> 分布式优化很好理解，因为数据量比较大以至于需要用多台机器来求解优化问题；统计学习就是能够通过计算充分统计量得到最优解的学习问题。 那么，**ADMM算法适用于大规模统计学习在分布式环境下的优化求解问题**。
> 
> 说明：本章ADMM相关，均源于此文和自己的理解。

**待解决的问题**

1. **为什么公式\\((diml.2.5.10)\\) (step2)中的迭代，用\\(\rho\\)作为step size** ? 

	对偶变量归一化（\\(\mu = \frac{1}{\rho} \beta\\)）后，能推导出\\(\mu^{k+1} \leftarrow \mu^{k} + \rho (w_t^{k+1} - \theta) \\).
	
2. \\(\rho\\)的物理意义？ 

	\\(\rho\\)仅仅是Augmented Lagrangian项的参数，因为对对偶变量做归一化，经过推导可以得到\\(\rho\\)是对偶变量更新步长。

### ADMM算法框架

交叉方向乘子法（Alternating Direction Method of Multipliers，简称ADMM）可以理解为是Augmented Lagrangians的变种。其为了整合Dual Ascent方法的可分解性和Augmented Lagrangians Multipliers优秀的收敛性质，进一步提出的新算法。

从ADMM名字上可以看出，它Augmented Lagrangians上添加了交叉方向（Alternating Direction），然后通过交叉换方向来交替优化。ADMM优化算法框架的结构形式如下：

$$
\begin{align}
& \min \quad f(x) + g(z) \\\
& s.b. \quad Ax + Bz = C
\end{align}  \qquad\qquad(diml.2.5.11)
$$

	
> 其中\\(x \in R^n, z \in R^m; A \in R^{p \times n}, B \in R^{p \times m}, C \in R^p\\)。
	
增强Lagrange函数

$$
\mathcal{L}_{\rho}(x,z,\beta) = f(x) + g(z) + \underline{ \beta^T(Ax+Bz-C) + \frac{\rho}{2} {\Vert Ax+Bz-C \Vert}_2^2 }  \qquad(diml.2.5.12)
$$
	
从上面形式确实可以看出，ADMM的思想就是想把primal变量、目标函数拆分，但是不再像dual ascent方法那样，将拆分开的\\(x_i\\)都看做是\\(\mathbf{x}\\)的一部分，后面融合的时候还需要融合在一起。而是最先开始就将拆开的变量分别看做是不同的变量\\(x\\)和\\(z\\)，约束条件也如此处理。这样的好处就是后面不需要一起融合\\(x\\)和\\(z\\)，保证了前面优化过程的可分解性。


<br>
#### 参数迭代形式


### 收敛性证明

### 一致性与信息共享

+ 问题求解的一致性，全局变量一致性
+ 全局变量共享

### ADMM与统计学习

| 算法框架 | 模型 | 参数学习方法 |
| --- | :--- | --- | 
| admm | Lasso <br> Logistic Regression <br> Factorization Machine <br> Filed-awared Factorization Machine | sgd <br> adaptive sgd <br> ftrl <br> lbfgs <br> mcmc |


### ADMM与机器通信

### ADMM应用场景

### ADMM算法应用

### 参考资料

### PPT

+ Precusors of ADMM
	+ 受限约束优化
		+ 原问题，拉格朗日，对偶问题（以SVM和最大熵模型为例），强对偶性以及成立条件 
	+ 对偶提升法
		+ 本质上是用梯度提升，求两个参数（x, beta），给出迭代公式；
		+ 之所以称为对偶提升，随着迭代进行，对偶函数是递增的。
		+ 梯度提升可用于对偶函数g不可导的场景，使用次梯度。
	+ 

