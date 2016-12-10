title: "第5章：DiMLSys-算法框架-ADMM" 
mathjax: true
date: 2015-01-05 18:30:31
categories: 
	- 分布式机器学习
tags: 
	- ADMM
	- 交替方向乘子法
	- 梯度提升法
	- 受限约束优化
---

+ author: zhouyongsdzh@foxmail.com
+ date: 2016-03-29
+ weibo: [@周永_52ML](http://weibo.com/p/1005051707438033/home?)

[站内跳转](#1)

**ADMM相关问题：**

1. ADMM算法的定位是什么？
2. ADMM算法为什么适合作为解决分布式机器学习任务的算法框架？ 
3. ADMM前世今生？（对偶提升，对偶分解，增强拉格朗日法）
4. ADMM算法框架的结构表达？
5. ADMM算法在分布式环境下的理论体系？（收敛性，一致性）
6. ADMM算法在分布式环境下工作过程？（以yarn为例）
7. 哪些学习模型可以适用于ADMM算法？
8. AFMM算法使用场景？

---

**内容列表**

+ 写在前面
+ 大规模机器学习的本质
+ 约束优化问题
	+ 对偶提升
	+ 对偶分解
	+ 增强拉格朗日乘子法
+ 交替方向乘子法
	+ 算法框架
	+ 收敛性证明
+ 一致性优化与均分优化
	+ 全局变量一致性优化（consensus）
	+ 均分优化（sharing）
+ ADMM在优化领域的具体应用
	+ 等式约束的凸优化问题
	+ $\ell_1 \text{norm}$问题 
+ ADMM与统计学习
	+ 支持的统计学习模型
	+ 预估模型
+ ADMM与机器通信
	+ ADMM与同步通信机制（以rabit为例）
	+ ADMM与异步通信机制（以ps为例）
+ ADMM应用场景
	+ 分布式机器学习任务（大数据学习）
	+ 多任务学习问题 
+ 参考资料

### 写在前面

前面的第2、3、4章内容主要论述了一个机器学习计算引擎需要依赖的底层系统，那么如何运用一个分布式算法框架让底层系统有机的运转，以实现大规模学习任务呢？

本章要论述了分布式优化算法框架－ADMM算法－来实现这一功能。ADMM算法并不是一个很新的算法，他只是整合许多不少经典优化思路，然后结合现代统计学习所遇到的问题，提出了一个比较一般的比较好实施的**分布式计算框架**。

ADMM算法结构天然地适用于分布式环境下具体任务的求解。在详细介绍ADMM分布式算法之前，我们先了解狭下一个大学习问题的如何在分布式环境下拆分成多个子任务学习问题的。然后通过《约束优化问题一般的解决方案》来阐述ADMM算法的演化过程，过渡到ADMM。最后详细阐述ADMM算法结构、理论可行性证明、分布式环境下如何保证一致性以及信息共享，适用的学习问题。

### 大规模机器学习的本质

大规模机器学习问题（large-scale machine learning）必须要解决“三大”问题：1. 大数据学习能力；2. 大模型学习能力；3. 产生更大的效果。

毫无疑问，大规模机器学习对比单机版机器学习任务，它要有更大的数据吞吐、处理和学习能力（远超出单机版能处理的数据范围）；由于存储和计算资源的增加，它必然得具体单机学习不了的复杂模型的能力，比如10亿级别以上的LR模型，更复杂的树模型等；在具备前两项能力的同时，必须要产生更大更好的效果（improve performance），否则大规模机器学习的存储毫无疑义，

> 业内其他的叫法，比如distributed machine learning，cloud machine learning，要表达的意思差不多.

因为是基于海量数据的学习任务，因此大规模机器学习解决方案基本上都需要将一个大的学习任务拆解成多个子任务并行（或分布式）进行：海量数据切分为若干块，每个子任务学习仅用一块数据进行训练，最后整合学习结果，得到最终参数解。公式化表示如下：

$$
\min_{w} \; \sum_{(x, y) \in \mathcal{D}} L(w^T x, y) + \lambda {\Vert w \Vert}_1 
\; \overset{\text{任务分解}}{\Longrightarrow} 
\min_{w_1, \cdots, w_T; w} \sum_{t=1}^{T} \left( \sum_{(x,y) \in \mathcal{D}_t} L(w_t^T x, y) \right) + \lambda {\Vert w \Vert}_1 \qquad(diml.2.5.0)
$$

$$
\min_{w} \; \sum_{(x, y) \in \mathcal{D}} L(w^T x, y) + \lambda {\Vert w \Vert}_1 
\xrightarrow[\text{ADMM结构}]{任务分解} 
\begin{align}
\min_{w_1, w_2, \cdots, w_T} & \; \sum_{t=1}^{T} \left( \sum_{(x,y) \in \mathcal{D}_t} L(w_t^T x, y) \right) + \lambda{\Vert \theta \Vert}_1  \\\
s.b. \quad & \; w_t=\theta \;(t=1,\cdots,T)
\end{align} 
$$

公式$(diml.2.5.0)$是一个带正则项的模型优化目标（```"损失函数＋正则项"```），损失函数用\\(L(w^T x, y)\\)表示。其中，$\mathcal{D}$表示训练集，把$\mathcal{D}$切分为$T$个子数据集，每一块用$\mathcal{D_t}$表示；模型参数$w \in R^n，n$为特征维度，$\lambda$为正则项系数。箭头右边是改写为多任务形式的优化目标，其中参数$w$起到了连接不同子任务的作用，$\lambda$控制了多个子任务连接的强度，$\lambda$越大说明连接强度越强。当$\lambda = 0$时，等价于$T$个子任务独立学习，子任务之间没有关联。

毫无例外地，任何一个大规模机器学习的任务都会用到“分而治之”的思想，即把大的机器学习任务拆分成多个子任务（大规模机器学习之间的差异主要在于拆分手段的不同）。不管如何拆分最后都会转化到一个本质问题上来，即多任务的联合学习。因此，可以把**大规模机器学习问题的本质是多任务的联合学习**。那么，接下来我们要思考的是如何求解该问题。

**问题1: 如何在分布式环境下求解多任务联合学习问题？** 

本章要讲述的ADMM算法就是解决这类学习问题不错的解决方案。严格意义上来讲，ADMM不同于梯度法、共轭梯度法、牛顿法、拟牛顿法等具体的参数学习算法，，把它称为**分布式计算框架**更合理。

在介绍ADMM算法之前，我们先来看下它的前身（precursors）是谁？是如何演化过来的？

### ADMM演化历程

说到ADMM算法的演化历程，要从等式约束优化问题说起。一个典型的等式约束优化问题，形式化表示如下：

$$
\begin{align}
& \; \min_{x} \;\; f(x) \\\
& s.t. \; Ax = b
\end{align} 	\qquad\qquad(diml.2.5.1)
$$

其中，**目标函数**\\(f(x): R^n \rightarrow R\\)，\\(Ax=b\\)为**约束条件**，参数\\(x \in R^n, A \in R^{m \times n}, b \in R^m\\)。\\(s.t\\)是英文```subject to```的缩写。如何求解等式约束优化问题？

<br>
#### 对偶提升法

等式约束优化问题一般是要用到拉格朗日乘子法。通过引入拉格朗日乘子，构造拉格朗日函数，分别对原参数和乘子求偏导（偏导数等于0），通过**交替优化**，使其最终收敛到最优解。其中，拉格朗日乘子又称算子、对偶变量。

公式\\((diml.2.5.1)\\)引入对偶变量（用\\(\beta \in R^m \\)表示），得到拉格朗日函数\\(\mathcal{L}: R^{m \times n} \rightarrow R \\) 为

$$
\begin{array}{lc}
& \; \min_{x} \;\; f(x) \\\
& s.t. \; Ax = b
\end{array}
\overset{拉格朗日函数}{\Longrightarrow}
\mathcal{L}(x, \beta) = f(x) + \beta^T (Ax-b)    \qquad\qquad(diml.2.5.2)
$$

把原问题的对偶问题表示为\\(\max \; g(\beta)\\)。对偶提升法的思想：在**强对偶性**假设下，通过最大化对偶函数，使得原函数和对偶函数会同时达到最优。求得最优解的过程是对参数的交替更新，参数更新的迭代公式如下：

$$
\begin{array}{lc}
& \; \min_{x} \;\; f(x) \\\
& s.t. \; Ax = b
\end{array}
\overset{拉格朗日函数}{\Longrightarrow}
\mathcal{L}(x, \beta) = f(x) + \beta^T (Ax-b)
\overset{对偶提升法}{\Longrightarrow}
\begin{align}
x^{k+1} & := \arg \min_{x} \mathcal{L}(x, \beta^{k})  \qquad\;(\text{step1}) \\\
\beta^{k+1} & := \beta^k + \alpha^{k}(A x^{k+1} - b) \\quad(\text{step2})
\end{align} \qquad(diml.2.5.3)
$$

收敛后得到原参数的最优解：

$$
x^{*} = \arg \min_{x} \mathcal{L}(x, \beta^{*})   \qquad\qquad\quad(diml.2.5.4)
$$

> 公式\\((diml.2.5.3)\\)解释：
> 
> step1: 求拉格朗日函数极小化时对应的参数\\(x\\)，具体实现时，需要对参数\\(x\\)求偏导，令\\(\frac{\partial}{\partial{x}} L(x, \beta^k) = 0\\).
> 
> step2: 公式\\((diml.2.5.3)\\)对\\(\beta\\)求偏导得到梯度，使用梯度提升法得到对偶变量\\(\beta\\)的迭代公式，\\(\alpha^k\\)为迭代步长。这一步称为**对偶提升(Dual Ascent)**。

**要想用对偶提升法求得最优解有一个前提假设：原目标函数必须是强凸函数**。否则得不到最优解，这也是对偶提升法的局限之处。

**问题2: 为什么对偶提升法对目标函数有强约束？**

参考[维基百科](https://en.wikipedia.org/wiki/Lagrange_multiplier)Figure2. 假设目标函数\\(y=f(x), x\\)是向量。\\(y\\)取不同的值，在\\(x\\)构成的平面（或曲面）上构成等高线。约束条件假设为\\(g(x) = c, x\\)是向量，在\\(x\\)构成的平面或去面上是一条曲线。

假设\\(g(x)\\)与等高线相交，交点就是同时满足等式约束条件和目标函数的可行域的值。但可以确定交点不是最优值，因为相交意味着肯定还存在其它的等高线在该条等高线的内侧或外侧，使得新的等高线与目标函数的交点的值更大或更小。只有当等高线与目标函数的曲线相切的时候，即目标函数的梯度方向与约束条件的梯度方向平行时缺德最优值，此时必须满足：\\(\nabla_{x} f(x) = \alpha \cdot \nabla_{x} g(x) \\)。而该式即为Lagrange函数\\(\mathcal{L}(x, \beta)\\)对参数\\(x\\)求偏导后的结果。

最优化领域的一些核心概念，如凸函数、对偶函数、共轭函数、强对偶性等，这里简单的说明如下：

> (1). 强凸函数
> 
> 在[《深入强出机器学习》系列 第10章：深入浅出ML之cluster家族]() 中的EM算法有提到.
> 强凸函数需满足：\\(E[f(x)] > f(E(x)) \\)
> 
> 函数\\(f: I \rightarrow R\\)成为强凸的，若\\(\exists\alpha > 0\\)，使\\(\forall(x, y) \in I \times I, \forall t \in [0, 1]\\)，恒有：
> 
$$
f[tx+(1-t)y] \le tf(x) + (1-t) f(y) - t(1-t) \alpha (x-y)^2 \qquad(n.diml.2.5.1)
$$

> (2). 对偶函数
> 
> 以公式$(diml.2.5.1)$等式约束优化问题为例，定义对偶函数\\(g(\beta): R^m \rightarrow R \\) 为Lagrange函数关于 \\(x\\)取得的最小值，即对\\(\beta \in R^m \\)有
$$
\begin{align}
g(\beta) & = \inf_{x} \mathcal{L}(x, \beta) = \inf_{x} \left(f(x) + \beta^T(Ax-b) \right) \\\
& = -\beta^T b + \inf_{x} \left(f(x) + \beta^TAx \right) = -\beta^T b - f^{*}(-A^T\beta)
\end{align} \qquad\\qquad(n.diml.2.5.2)
$$
> 其中\\(f^{*}\\)是\\(f\\)的[共轭函数](参考wiki)。
>
> (3). 共轭函数
> 
> 设函数\\(f: R^n \rightarrow R\\)，定义函数\\(f^{*}: R^n \rightarrow R\\)为 
$
f^{*} = \inf_{x \in dom \, f} \left(f(x) - y^T x \right)
$
> 。此函数称为函数\\(f\\)的共轭函数。（参考《Convex Optimization》3.3节 共轭函数定义.）

> (4). 强对偶性
> 
> 在约束优化问题中，如果最小化**原凸函数**等价于最大化**对偶函数**时，称为强对偶性。即\\(\min f(x) = \max g(\beta)\\)
>
对偶提升法在满足强对偶性假设下可以证明公式\\((diml.2.5.3)\\)能达到收敛。即要求目标函数\\(f(x)\\)为强凸函数。

对偶提升法虽然对目标函数有严格的约束，使得很多优化目标不能用其求解。但是它具体一个很好的性质，详细的见下面要提到的对偶分解。


<br>
#### 对偶分解

对偶提升法虽然对目标函数有严格的要求，但是它还有一个非常好的性质：

**如果目标函数\\(f(x)\\)是可分的，整个优化问题可以拆分成多个子优化问题，分块优化后得到局部分数，然后汇集起来 整体更新全局参数，有利于问题的并行化处理。这个过程称为对偶分解（Dual Decomposition）。**

如果目标函数\\(f(x)\\)和约束条件是可分解的，那么原问题\\((diml.2.5.1)\\)对应的分解形式为：

$$
\begin{array}{lc}
& \; \min_{x} \;\; f(x) \\\
& s.t. \; Ax = b
\end{array}
\overset{分解原问题}{\Longrightarrow}
\begin{align}
& \min_{x} \; f(x) = \sum_{t=1}^{T} f_t(x_t) \\\
& s.t. \; Ax = \sum_{t=1}^{T} A_t x_t = b 
\end{align}   \qquad\qquad(diml.2.5.5)
$$

拉格朗日函数 与 参数更新公式：

$$
\mathcal{L}(x, \beta) = \sum_{t=1}^{T} \mathcal{L_t}(x_t, \beta) = \sum_{t=1}^{T} \left(f_t(x_t) + \beta^T A_t x_t - \frac{1}{T}\beta^T b \right) 
\Longrightarrow 
\begin{array}{lc}
x_{t}^{k+1} := \arg \min_{x} \mathcal{L}_t(x_t,\beta^{k}) \qquad\qquad\qquad\quad(\text{1})\\\
\beta^{k+1} := \beta^{k} + \alpha^k \nabla g(\beta) = y^k + \alpha^k(A x^{k+1} -b)  \;\;\,(\text{2})
\end{array} \;(diml.2.5.7)
$$

> 公式$(diml.2.5.7)$解读：
> 
> step1: 并行化求解多个子目标函数
> 
> step2: 汇总\\(x_{i}^{k+1}\\)（论证是否求平均可否？），更新对偶变量.

如何在目标函数不满足强凸函数约束时，求解对应的优化问题呢？下面要提到的增广拉格朗日乘子法可以解决。

<br>
#### 增广拉格朗日乘子法

前面提到，对偶提升方法求解优化问题时，目标函数必须满足强凸函数的条件，限制过于严格。为了增加对偶提升法的鲁棒性和放松对目标函数\\(f\\)强凸的约束条件，人们提出了[增广拉格朗日乘子法](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method)（Augmented Lagrangians）用于解决这类问题。

Argumented Lagrangians方法的思想：**在目标函数基础上引入惩罚函数项（二次项），放松对目标函数\\(f(x)\\)严格凸的限制，同时使得算法更加稳健。**


最优化问题从\\((diml.2.5.1)\\)变为：

$$
\begin{array}{lc}
& \; \min_{x} \;\; f(x) \\\
& s.t. \; Ax = b
\end{array}
\overset{惩罚函数项}{\Longrightarrow}
\begin{align}
& \; \min_{x} \;\; f(x) + \frac{\rho}{2} {\Vert Ax-b \Vert}_2^2 \\\
& \quad s.t. \; Ax = b
\end{align} 	\qquad\qquad(diml.2.5.8)
$$

在约束条件\\(Ax-b=0\\)下，目标函数定义为\\(f(x) + \frac{\rho}{2} {\Vert Ax-b \Vert}_2^2\\) 与\\(f(x)\\)是等价的，因为最优解在满足约束条件的前提下优化目标的后一项等于0，这一项称为**惩罚函数项**（或二次惩罚项）。

对应的拉格朗日函数：

$$
\mathcal{L}_{\rho}(x, \beta) = f(x) + \frac{\rho}{2} {\Vert Ax-b \Vert}_2^2 + \beta^T (Ax-b)    \qquad\quad(diml.2.5.9)
$$


> 公式解读：
> 
>
$$
\mathcal{L}_{\rho}(x, \beta) = \overbrace{f(x)}^{原优化目标} + \overbrace{\underbrace{\frac{\rho}{2} {\Vert Ax-b \Vert}_2^2}_{二次惩罚项} + \underbrace{\beta^T(Ax-b)}_{拉格朗日乘子项} }^{增广拉格朗日乘子项}  \qquad (n.diml.2.5.3)
$$



**问题3：为什么添加二次惩罚项就可以“解除对目标函数\\(f(x)\\)强凸性质”的限制, 进而可得最优解了呢？**

原因在于如果**原优化目标\\(f(x)\\)是非凸的，那意味着拉格朗日函数在对\\(x\\)求偏导时 结果可能不可导，无法进行参数交替迭代优化。添加了二次项惩罚项之后，可以保证即便\\(f(x)\\)不满足强凸条件，在对x求偏导时 仍然可导，因为二次惩罚项部分可以保证这一点**。

因此，添加二次惩罚项的好处是可以保证增广的对偶函数一定是可微的。迭代公式如下：

$$
\mathcal{L}_{\rho}(x, \beta) = f(x) + \frac{\rho}{2} {\Vert Ax-b \Vert}_2^2 + \beta^T (Ax-b) 
\Longrightarrow 
\begin{align}
x^{k+1} & := \arg \min_{x} \mathcal{L}_{\rho} (x, \beta^k) \qquad (step1) \\\
\beta^{k+1} & := \beta^k + \rho(Ax^{k+1}-b)  \quad\;\;(step2)
\end{align}  \qquad(diml.2.5.10)
$$

上式即为增广拉格朗日乘子法的交替迭代公式。与标准的对偶提升法类似，但有两点不同：

1. $\text{step1}$. \\(\mathcal{L}(x, \beta)\\)对\\(x\\)求偏导得到最小值时用的是增广拉格朗日函数；
2. $\text{step2}$. 惩罚项参数\\(\rho\\)在这里作为对偶变量更新时的步长（step size）。

**问题4：为什么选择\\(\rho\\)作为对偶变量更新的迭代的step size？**

为了推导方便，这里假设\\(f\\)是可微的（尽管在Augmented拉格朗日法中不是必须的），\\((x^{\star}, \beta^{\star})\\)是优化问题的最优解。此时应该满足以下条件：

$$
Ax^{\star} - b = 0,\; \nabla f(x^{\star}) + A^T\beta^{\star} = 0 \qquad (diml.2.5.11)
$$

根据参数交替迭代公式的\\(step1\\)，第\\(k+1\\)的迭代结果\\(x^{k+1}\\)是由最小化函数\\(\mathcal{L}_{\rho}(x, \beta^k)\\)得来（具体做法：对参数\\(x\\)求偏导，令偏导数等于0），所以\\(x^{k+1}\\)满足下式成立

$$
\begin{align}
0 & = \nabla_{x} \mathcal{L}_{\rho}(x^{k+1}, \beta^k) = \nabla_{x} f(x^{k+1}) + \beta^k A + \rho( A^T Ax - A^T b) \\\
& = \nabla_{x}f(x^{k+1}) + A^T \left(\underline {\beta^k + \rho(Ax^{k+1} - b) } \right) \\\
& = \nabla_{x}f(x^{k+1}) + A^T \beta^{k+1}
\end{align} \qquad(diml.2.5.12)
$$

因此可以得到，\\(\rho\\)为对偶变量参数更新的步长，并得到了第\\(k+1\\)次迭代的最优解\\((x^{k+1}, \beta^{k+1})\\)。随着迭代的进行，原函数的残差\\(Ax^{k+1}-b\\)逐步收敛到0，得到最优解。

相比对偶提升法，增广拉格朗日乘子法有更好的收敛性质，并且拥有对目标\\(f(x)\\)不做强凸条件的限制，但是这些好处总是要付出一定的代价的：**如果目标函数是\\(f(x)\\)是可分的，此时增广拉格朗日函数\\(\mathcal{L}_{\rho}(x, \beta)\\)是不可分的（因为惩罚函数项部分涉及到矩阵相乘计算，无法用分块形式进行并行化求解），因此公式\\((diml.2.5.10)\\)的\\(step1\\)没有办法在分布式环境下并行优化**。

如果能结合Dual Ascent的并行求解的优势 和 Augmented Lagrangians Methods of Multipliers鲁棒性和不错的收敛性质，那么就可以使大多数目标函数求解都能在分布式环境下并行实现，岂不美哉！！！

果然在2010年由Stephen Boyd大师等人系统性的整理了交替方向乘子法（ADMM算法），可以解决上述的问题。为此它们长篇论述了ADMM算法的演化过程，参考论文：[Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers](http://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf) 该文被《Foundations and Trends in Machine Learning》录入。

### 交替方向乘子法

交替方向乘子法（Alternating Direction Method of Multipliers，简称ADMM）**适用于大规模统计学习在分布式环境下的优化求解问题**。可以理解为增广拉格朗日乘子法的变种，旨在整合对偶提升法的可分解性和增广拉格朗日乘子法优秀的收敛性质，进一步提出的新算法。

#### ADMM算法框架

我们修改下公式\\((diml.2.5.1)\\)，这样更符合统计学习目标函数的形式. 重新定义的优化问题和Lagrange函数：

$$
\begin{array}{lc}
\min_{x} \;\; f(x) \\\
s.t. \; Ax = b
\end{array}
\overset{目标函数分解}{\Longrightarrow}
\begin{array}{lc}
\min & f(x) + g(z) \\\
s.t. & Ax + Bz = C
\end{array}  \qquad\quad \quad (diml.2.5.13) 
$$


其中\\(x \in R^n, z \in R^m; A \in R^{p \times n}, B \in R^{p \times m}, C \in R^p\\)。

拉格朗日函数：

$$
\mathcal{L}_{\rho}(x, z, \beta) = f(x) + g(z) +\underline{ \frac{\rho}{2} {\Vert Ax+Bz-C \Vert}_2^2 + \beta^T(Ax+Bz-C) } \quad (diml.2.5.14) 
$$

按照乘子法的思路，参数交替更新迭代：

$$
\begin{align}
(x^{k+1}, z^{k+1}) & := \arg \min_{x,z} \mathcal{L}_{\rho}(x,z,\beta^k) \qquad\qquad(\text{step1}) \\\
\beta^{k+1} & := \beta^k + \rho (Ax^{k+1} + Bz^{k+1} - C) \quad(\text{step2})
\end{align}  \qquad(diml.2.5.15)
$$

上式中的\\(step1\\)要求对两个原始变量联合最小化，也就是说\\(x,z\\)是融合在一起优化的，暂且不说联合优化是否容易求解，可以确定的这一步优化不可分解。

ADMM采用了拆分思想，最初就把\\(x\\)和\\(z\\)分别看作两个不同的变量，约束条件也是如此。采用交替方式迭代（序贯式迭代），称为**交替方向**(alternating direction)。

$$
\begin{align}
x_t^{k+1} & := \arg \min_x \mathcal{L}_{\rho}(x, z^{k}, \beta^k) \qquad\qquad (\text{step1, 局部更新}) \\\
z^{k+1} & := \arg \min_z \mathcal{L}_{\rho}(x^{k+1}, z, \beta^k) \qquad\quad\; (\text{step2, 全局更新}) \\\ 
\beta_t^{k+1} & := \beta^{k} + \rho(Ax^{k+1} + Bz^{k+1} - C) \quad\;\; (\text{step3, 局部更新})
\end{align}  \qquad\qquad(diml.2.5.16)
$$

ADMM算法拆分参数\\(x\\)和\\(z\\)两步迭代最大的好处是：**当\\(f\\)可分时，参数可以并行求解。**

在[```chapter6_DiML_算法框架_学习器```]()中可以看到，ADMM这种参数和目标函数的拆分非常适合机器学习中的\\(\ell_1 \text{-norm}\\)优化问题，即：```loss function + regularization```目标函数的分布式求解。


#### ADMM算法性质与评价

**1). 收敛性**

论文中对收敛性的证明，提到了两个假设条件：

+ $f(x)$和$g(z)$分别是扩展的实质函数：\\(R^{n}(R^{m}) \rightarrow R \; \cup {+\infty} \\), 并且是closed、proper和convex的；
+ 增广拉格朗日函数\\(\mathcal{L}_0\\)有一个鞍点（saddle point）；对于约束中的矩阵$A,B$都不需要满秩。

满足两个假设条件下，可以保证残差、目标函数、对偶变量的收敛性。（详细证明过程参考paper Appendix A）.

> 实际应用表明，ADMM算法收敛速度是很慢的，类似于共轭梯度法。迭代数十次可以得到一个可接受的结果，与快速的高精度算法（牛顿法、拟牛顿法、内点法等）相比收敛就满多了。因此实际应用中ADMM会与其它高精度算法结合其俩，这样从一个可接受的结果变得在预期时间内可以达到较高的收敛精度。
> 
> 在大规模问题求解中，高精度的参数解对于预测的泛化效果没有很大的提高。因此实际应用中，预期时间内得到的一个可接受的结果就可以直接应用预测了。


**2). 最优条件和停止准则**

最优条件先省略。放在公式(diml.2.5.17)中

从最优条件中可以得到初始残差（primal residuals）和对偶残差（dual residuals）的表达式：

$$
\begin{align}
r^{k+1} & := Ax^{k+1} + Bz^{k+1} - C \qquad(初始残差) \\\
s^{k+1} & := \rho A^T B(z^{k+1} - z^{k}) \qquad\quad(对偶残差)
\end{align}  \qquad\qquad(diml.2.5.18)
$$

迭代停止准则比较难以把握，因为受收敛速度问题，要想获得一个不错的参数解，判断迭代停止条件还是比较重要的。实际应用中，一般都根据初始残差和对偶残差足够小来停止迭代。阈值包含了绝对容忍度（absolute tolerance）和相对容忍度（relative tolerance），阈值设置难以把握，具体形式如下：

$$
\begin{align}
{\Vert s^k \Vert} \leq \epsilon^{\text{dual}} & = \sqrt{n} \epsilon^{\text{abs}} + \epsilon^{\text{rel}} {\Vert A^T y^k \Vert}_2 \\\
{\Vert r^k \Vert}_2 \leq & = \sqrt{p} \epsilon^{\text{abs}} + \epsilon^{\text{rel}} \; \text{max} \{ {\Vert Ax^k \Vert}_2, {\Vert Bz^k \Vert} _2, {\Vert C \Vert}_2 \}
\end{align}  \qquad\quad (diml.2.5.19)
$$

上面的$\sqrt{p}$和$\sqrt{n}$分别是维度和样本量。一般而言，相对停止阈值$\epsilon^{\text{rel}} = 10^{-1}$或$10^{-4}$，绝对阈值的选取要根据变量的取值范围来选取。

此外，在对偶变量更新的惩罚参数\\(\rho\\)原来是不变的。有一些文章做了可变的惩罚参数，目的是为了降低惩罚参数对初始值的依赖。而证明变动的$\rho$给ADMM的收敛性证明比较困难，因此实际中开设经过一系列迭代后$\rho$也稳定，直接用固定的惩罚参数$\rho$了。

### 一致性优化和均分优化

分布式优化有两个很重要的问题，即一致性优化问题和共享优化问题，这也是ADMM算法通往并行和分布式计算的途径。下面以机器学习问题的参数优化为例解释这两个重要的概念。

**全局变量一致性优化（global consensus）**

所谓全局变量一致性优化问题，说白了就是**切割大样本数据，并行化计算**。即整体优化求解任务根据目标函数分解成$T$个子任务，每个子任务和子数据都可以获得一个参数解$x_i$，但是整体优化任务的解只有一个$z$。机器学习中的优化目标通常是“损失函数＋正则项”的结构形式，问题就转化为**带正则项的全局一致性问题**，优化问题可以写成如下形式：

$$
\begin{array}{lc}
\min \quad \sum_{i=1}^{T} \; f_i(x_i) + g(z)\\\
s.t. \quad x_i = z, \; x_i \in R^n, i=1,\cdots,T
\end{array}
\overset{参数迭代}{\Longrightarrow }
\begin{align}
x_i^{k+1} & := \arg \min_{x_i} \left(f_i(x_i) + (\beta_i^{k})^T (x_i - z^k) + \frac{\rho}{2} {\Vert x_i - z \Vert}_2^2 \right) \\\
z^{k+1} & := \arg \min_{z} \left(g(z) - \sum_{t=1}^{T} (\beta_t^k)^T \theta + \frac{\rho}{2} {\Vert x_i - z \Vert}_2^2 \right) \\\
\beta_i^{k+1} & := \beta_i^k + \rho (x_i^{k+1} - z^{k+1})
\end{align}
$$

注意，此时$f_i : R^n \rightarrow R \, \cup {+\infty}$仍然是凸函数，而局部参数$x_i$并不是对参数空间进行划分，而是把数据集合$\mathcal{D}$划分为$T$个子集，对数据划分。下文提到的ADMM在$\ell_{1}\text{-norm}$问题中具体应用会详细推导这部分。

**均分优化问题（sharing）**

在统计学习优化问题中，所谓均分优化问题说白了就是**切分特征维度，并行化求解**。它更侧重于高维数据并行化求解的应用场景，通过切分特征维度得到并行化处理。同样假设数据矩阵$A \in R^{m \times n}$和$b \in R^m$，此时满足$n \gg m$，也就是说样本特征维度远大于样本数。shareing的做法是对数据矩阵$A$按照特征空间切分（竖着切分），

$$
\begin{array}{lc}
A = [A_1, A_2, \cdots, A_T], A_i \in R^{m \times n_i} \\\ 
x = (x_1, x_2, \cdots, x_T), x_i \in R^{n_i} 
\end{array}
\overset{数据和特征按列切分}{\Longrightarrow}
\begin{array}{lc}
Ax = \sum_{i=1}^{T} A_i x_i ; \\\
g(x) = \sum_{i=1}^{T} g_i(x_i)
\end{array}
\Longrightarrow 
\min \; L \left(\sum_{i=1}^{T} A_i x_i - b \right) + \sum_{i=1}^{T} g_i(x_i)  \quad(diml.2.5.21)
$$

我们把公式$(diml.2.5.21)$改成ADMM sharing形式：

$$
\begin{array}{lc}
\min \; L(\sum_{i=1}^{T} z_i - b) + \sum_{i=1}^{T} g_i(x_i) \\\
s.t. \quad A_i x_i - z_i = 0, \; z_i \in R^m
\end{array}
\Longrightarrow 
\begin{align}
x_i^{k+1} & := \arg \min_{x_i} \left(g_i(x_i) + \frac{\rho}{2} {\Vert A_i x_i - A_i x_i^k + \overline{Ax}^k - \overline{z}^k + \mu^k \Vert}_2^2 \right) \\\
z^{k+1} & := \arg \min_{z} \left(L(N\overline{z} - b) + T \frac{\rho}{2} {\Vert \overline{z} - \overline{Ax}^{k+1} - \mu^k \Vert}_2^2 \right) \\\
\mu_{k+1} & := \mu_i^k + \overline{Ax}^{k+1} - \overline{z}^{k+1}
\end{align} 
$$

### ADMM在优化领域的具体应用

#### $\ell_{1}\text{-norm}$问题

 这里的$\ell_{1}\text{-norm}$问题不仅仅是指称为[Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics))问题，而是包含了多种$\ell_{1}\text{-norm}$类型问题。 它的初衷是通过特征选择（自变量选择）来提高模型的预测精度和解释性，具体做法是在优化目标上添加$\ell_{1}$正则项。

> Lasso 的基本思想是在回归系数的绝对值之和小于一个常数的约束条件下，使残差平方和最小化，从而能够产生某些严格等于0 的回归系数，得到可以解释的模型。

如果要大规模部署$\ell_{1}\text{-norm}$问题的解决方案，ADMM算法可能是首选。它非常适合机器学习和统计学习的优化问题，因为机器学习问题的优化目标大部分都是**“损失函数＋正则项”**形式. 这种表达形式切好饿意套用ADMM算法框架$f(x) + g(z)$。因此机器学习问题结合ADMM算法框架基本可以在分布式环境下求解很多已由的问题。

本系列我们用ADMM算法框架主要解决分布式机器学习任务的参数优化问题，所以我们直接关注**一般化的损失函数＋$\ell_{1}正则项问题$**。这类问题通用框架：

以平方损失函数为例，定义损失函数\\(L(y^{(i)}, w^Tx^{(i)}) = \left(y^{(i)} - w^Tx^{(i)} \right)^2\\). 优化目标：

$$
\begin{array}{lc}
\min_{w} \quad \frac{1}{m} \sum_{i=1}^{m} \left(y^{(i)} - w^Tx^{(i)} \right)^2 + \lambda {\Vert w \Vert}_1
\end{array} \qquad\qquad\qquad\qquad\quad (整体优化目标)\\\ . \\\
\overset{\text{ADMM}形式} {\Longrightarrow } \quad 
\begin{array}{lc}
\min_{w_1, w_2, \cdots, w_T} \quad \sum_{t=1}^{T} \left( \frac{1}{m_t} \sum_{i=1}^{m_t} \left(y_t^{(i)} - w_t^T x_t^{(i)} \right)^2 \right) + \lambda{\Vert \theta \Vert}_1  \\\
s.b \qquad\; w_t=\theta \;(t=1,\cdots,T)
\end{array} \quad (分布式优化目标表达式)
$$

令\\(\frac{1}{m_t} \sum_{i=1}^{m_t} \left(y_t^{(i)} - w_t^T x_t^{(i)} \right)^2 = L(\mathbf{y}_t, \mathbf{x}_t w_t), \; \\)，增广拉格朗日函数：

$$
\mathcal{L}(w_t, \theta, \beta_t) = \sum_{t=1}^{T} \; L(\mathbf{y}_t, \mathbf{x}_t w_t) + \lambda {\Vert \theta \Vert}_1 + \sum_{t=1}^{T} \beta_t^T(w_t - \theta) + \frac{\rho}{2} \sum_{t=1}^{T} {\Vert w_t - \theta \Vert}_2^2
$$

**参数迭代过程**

+ 局部参数更新（worker节点）

$$
\begin{align}
w_t^{k+1}  \longleftarrow \; & \arg\min_{w_t} \; \frac{1}{m_t}\; L(\mathbf{y}_t, \mathbf{x}_t w_t) + (\beta_t^k)^T w_t + \frac{\rho}{2} {\Vert w_t - \theta^k \Vert}_2^2 \qquad\qquad\quad(1) \\\
\; & \arg\min_{w_t} \; \frac{1}{m_t}\; L(\mathbf{y}_t, \mathbf{x}_t w_t) + \frac{\rho}{2} {\Vert w_t - \theta^k + \frac{1}{\rho} \beta_t^k \Vert}_2^2
\end{align}
$$

+ 全局参数更新（master节点）

	$$
\theta^{k+1} \longleftarrow \arg\min_{\theta} \; \lambda {\Vert \theta \Vert}_1 - \sum_{t=1}^{T} (\beta_t^k)^T \theta + \frac{\rho}{2} \sum_{t=1}^{T} {\Vert w_t^{k+1} - \theta \Vert}_2^2 \qquad\qquad(2)
	$$

+ 对偶变量更新（worker节点）

	$$
\begin{align}
\beta_t^{k+1} \longleftarrow & \beta_t^k + \rho(w_t^{k+1} - \theta^{k+1}) \qquad\qquad\qquad\qquad\qquad\qquad\qquad\;\,(3)
\end{align}  
$$

迭代公式说明：

+ 第1步：局部参数更新。目标函数可以看作是**```损失函数+L2正则项```**(\\({\Vert w_t - const \Vert}_2^2\\))，涉及参数：$(\theta, \beta_t, w_t)$；
+ 第2步：全局参数更新。需要详细推到，涉及到软阈值. 涉及参数：\\((w_1,\cdots,w_T, \rho, \beta_1,\cdots,\beta_T,\lambda)\\)
+ 第3步：局部对偶变量更新。涉及参数：\\((\theta, \lambda, w_1,\cdots,w_T, \rho)\\)

这里涉及到一个worker节点参数与master节点参数通信问题，以及**master节点是如何利用worker局部参数更新全局参数的？**我们先看master节点参数更新推导过程。

拉格朗日函数，对\\(\theta\\)求偏导：

$$
\begin{align}
\frac{\partial \, \mathcal{L}(w_t, \theta, \beta_t)} {\partial{\theta}} & ＝ \frac{\partial \; \left({\lambda {\vert \theta \vert}_1} - \sum_{t=1}^{T}(\beta_t)^T \theta + \frac{\rho}{2} \sum_{t=1}^{T} {\Vert w_t - \theta \Vert}_2^2 \right)} {\partial {\theta}} \\\
& = \mathbf{sign}(\theta) \cdot \lambda - \sum_{t=1}^{T} \beta_t + {\rho} \sum_{t=1}^{T} \left(\theta - w_t \right) \\\
& = \mathbf{sign}({\vert \theta \vert}_1) \cdot \frac{\lambda}{\rho} - \sum_{t=1}^{T} \left( \frac{\beta_t}{\rho} + w_t\right) + \sum_{t=1}^{T} \theta = 0 \\\
\end{align}
$$

$$
全局参数\theta更新公式整理得到：\underline{\color{blue}{ \theta } = \frac{1}{T} \left(\sum_{t=1}^{T} 
\left(\color{red} { \frac{\beta_t}{\rho} + w_t } \right)  - sign({\vert \theta \vert}_1) \cdot \frac{\lambda}{\rho} \right) }
$$

说明：全局参数更新时，$\ell_{1}\text{-norm}$项需要求导。虽然在0处不可导，但仍有解析解，这里使用**[软阈值]()**的方法得到解析解：

$$
\color{blue}{\theta} =
\begin{cases}
\frac{1}{T} \left( \sum_{t=1}^{T} \left( \color{red}{\frac{\beta_t}{\rho} + w_t}\right) + \frac{\lambda}{\rho}  \right) & \qquad \text{if} \; {\vert \theta \vert}_1 < 0, 若：\sum_{t=1}^{T} \left( \frac{\beta_t}{\rho} + w_t\right) + \frac{\lambda}{\rho} < 0 ; \\\
\frac{1}{T}\left(\sum_{t=1}^{T}\left(\color{red}{\frac{\beta_t}{\rho} + w_t} \right) - \frac{\lambda}{\rho} \right) & \qquad \text{if} \; {\vert \theta \vert}_1 > 0, 若：\sum_{t=1}^{T} \left( \frac{\beta_t}{\rho} + w_t\right) - \frac{\lambda}{\rho} > 0 ; \\\
\; 0  & \qquad \text{otherwise}.
\end{cases}
$$

参数：\\(\lambda\\) 为$L_1$正则项系数；\\(\beta_t\\)对偶变量（拉格朗日乘子）; \\(\theta\\)全局参数; \\(w_t\\)局部参数;

公式中的红色区域是worker向master传递的传递参数，这一步称为merge过程（在MPI对应allreduce操作，Hadoop对应reduce过程）；蓝色区域（全局参数）是master向所有


> [软阈值（Soft-Thresholding）]()又称压缩算子（shrinkage operator）
> 


#### 受约束的凸优化问题



<br>
### ADMM for CTR Model
--

generalized lasso, group lasso, 高斯图模型，Tensor型图模型等问题的求解，都可以在ADMM算法框架上直接应用和实施，这正是ADMM算法的一个优势所在，便于大规模分布式部署。

| 算法框架 | 模型 | 参数学习方法 |
| --- | :--- | --- | 
| admm | Lasso <br> Logistic Regression <br> Factorization Machine <br> Filed-awared Factorization Machine | sgd <br> adaptive sgd <br> ftrl <br> lbfgs <br> mcmc |

### ADMM应用场景

+ Big-Data Learning 
+ Multi-Task Learning 

### ADMM算法应用

### 参考资料

+ [分布式计算、统计学习与ADMM算法](http://joegaotao.github.io/cn/2014/02/admm/)

<h2 id = "1">持续更新中</h2>

