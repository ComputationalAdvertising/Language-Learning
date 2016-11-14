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
+ weibo: [@周永_52diml](http://weibo.com/p/1005051707438033/home?)

---

**内容列表**

+ 写在前面
+ 分布式环境下的机器学习问题
+ 约束优化问题
	+ 对偶提升
	+ 对偶分解
	+ 增强拉格朗日乘子法
+ 交替方向乘子法
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

前面的第2、3、4三章内容主要论述了一个完整的机器学习系统的底层系统，那么如何在算法层面让底层系统有机的运转，以实现分布式学习任务呢？

本章要论述了分布式优化算法框架－ADMM算法－来实现这一功能。ADMM算法并不是一个很新的算法，他只是整合许多不少经典优化思路，然后结合现代统计学习所遇到的问题，提出了一个比较一般的比较好实施的**分布式计算框架**。

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

### ADMM从哪里来

谈到ADMM算法的演化历程，应该要从等式约束优化问题说起。一个典型的等式约束优化问题，形式化表示如下：

$$
\begin{align}
& \; \min_{x} \;\; f(x) \\\
& s.t. \; Ax = b
\end{align} 	\qquad\qquad(diml.2.5.1)
$$

其中，**目标函数**\\(f(x): R^n \rightarrow R\\)，\\(Ax=b\\)为**约束条件**，参数\\(x \in R^n, A \in R^{m \times n}, b \in R^m\\)。\\(s.t\\)是英文```subject to```的缩写。如何求解这个等式约束优化问题呢？

$$
\begin{array}{lc}
\min & f(x) \\\
s.t. & Ax = b \\
\end{array}
\Longrightarrow L(x, y) = f(x) + y^T(Ax - b) \overset{对偶函数（下界）}{\Longrightarrow} g(y) = \inf_x L(x, y)
$$

<br>
#### 对偶提升法

拉格朗日乘子法通过引入拉格朗日乘子，构造拉格朗日函数，分别对原参数和乘子求偏导且让其等于0，通过**交替优化**，使其最终收敛到最优解。其中，拉格朗日乘子又称算子、对偶变量。

我们对公式\\((diml.2.5.1)\\)引入对偶变量，用\\(\beta \in R^m \\)表示，得到拉格朗日函数\\(\mathcal{L}: R^{m \times n} \rightarrow R \\) 为

$$
\mathcal{L}(x, \beta) = f(x) + \beta^T (Ax-b)    \qquad\qquad(diml.2.5.2)
$$

对偶函数\\(g(\beta): R^m \rightarrow R \\) 为Lagrange函数关于\\(x\\)取得的最小值：即对\\(\beta \in R^m \\)，有

$$
\begin{align}
g(\beta) & = \inf_{x} \mathcal{L}(x, \beta) = \inf_{x} \left(f(x) + \beta^T(Ax-b) \right) \\\ 
& = -\beta^T b + \inf_{x} \left(f(x) + \beta^TAx \right) \\\
& = -\beta^T b - f^{*}(-A^T\beta)
\end{align} \qquad\\qquad(diml.2.5.3)
$$

其中\\(f^{*}\\)是\\(f\\)的[共轭函数](参考wiki)。

> 共轭函数定义：
> 
> 设函数\\(f: R^n \rightarrow R\\)，定义函数\\(f^{*}: R^n \rightarrow R\\)为
> 
$$
f^{*} = \inf_{x \in dom f} \left(f(x) - y^T x \right)
$$
>
> 此函数称为函数\\(f\\)的共轭函数。（参考《Convex Optimization》3.3节 共轭函数定义.）

原问题的对偶问题为\\(\max \; g(\beta)\\)。对偶提升法就是通过最大化对偶函数，在**强对偶性**假设下，原函数和对偶函数会同时达到最优，可以得到：

$$
x^{*} = \arg \min_{x} \mathcal{L}(x, \beta^{*})   \qquad\qquad\quad(diml.2.5.4)
$$

> 什么是强对偶性？
> 
> 如果最小化**原凸函数**等价于最大化**对偶函数**时，称为强对偶性。即\\(\min f(x) = \max g(\beta)\\)

求得最优解的过程，是通过交替更新参数，使其同时收敛到最优。参数更新的迭代公式如下：

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
> step2: 公式\\((diml.2.5.3)\\)对\\(\beta\\)求偏导得到梯度，使用梯度提升法得到对偶变量\\(\beta\\)的迭代公式，\\(\alpha^k\\)为迭代步长。这一步称为对偶提升(Dual Ascent)。

对偶提升法在满足强对偶性假设下可以证明公式\\((diml.2.5.5)\\)能达到收敛。即要求目标函数\\(f(x)\\)为强凸函数。

> 什么是强凸函数？（在[《深入强出机器学习》系列 第10章：深入浅出ML之cluster家族]() 中的EM算法有提到）
> 
> 强凸函数需满足：\\(E[f(x)] > f(E(x)) \\)
> 
> 函数\\(f: I \rightarrow R\\)成为强凸的，若\\(\exists\alpha > 0\\)，使\\(\forall(x, y) \in I \times I, \forall t \in [0, 1]\\)，恒有：
> 
$$
f[tx+(1-t)y] \le tf(x) + (1-t) f(y) - t(1-t) \alpha (x-y)^2
$$

**为什么对偶提升法对目标函数有较强的约束？**

参考[维基百科](https://en.wikipedia.org/wiki/Lagrange_multiplier)Figure2. 假设目标函数\\(y=f(x), x\\)是向量。\\(y\\)取不同的值，在\\(x\\)构成的平面（或曲面）上构成等高线。约束条件假设为\\(g(x) = c, x\\)是向量，在\\(x\\)构成的平面或去面上是一条曲线。

假设\\(g(x)\\)与等高线相交，交点就是同时满足等式约束条件和目标函数的可行域的值。但可以确定交点不是最优值，因为相交意味着肯定还存在其它的等高线在该条等高线的内侧或外侧，使得新的等高线与目标函数的交点的值更大或更小。只有当等高线与目标函数的曲线相切的时候，即目标函数的梯度方向与约束条件的梯度方向平行时缺德最优值，此时必须满足：\\(\nabla_{x} f(x) = \alpha \cdot \nabla_{x} g(x) \\)。而该式即为Lagrange函数\\(\mathcal{L}(x, \beta)\\)对参数\\(x\\)求偏导后的结果。

---

3. Augmented Lagrangian法添加二次项后 为什么可以缓解约束？
4. 为什么Augmented Lagrangian不能分解 ？是参数不能分解还是二次项不能分解？
5. ADMM算法框架为什么可以分解？

---

<br>
#### 对偶分解

对偶提升法虽然对目标函数有严格的要求，但是它还有一个非常好的性质：

**如果目标函数\\(f(x)\\)是可分的，整个优化问题可以拆分成多个子优化问题，分块优化后得到局部分数，然后汇集起来 整体更新全局参数，有利于问题的并行化处理。这个过程称为对偶分解（Dual Decomposition）**

目标函数\\(f(x)\\)如果可分，可以对原问题\\((diml.2.5.1)\\)拆分为：

$$
\begin{align}
& \min_{x} \; f(x) = \sum_{i=1}^{k} f_i(x_i) \\\
& s.t. \; Ax = \sum_{i=1}^{k} A_i x_i = b 
\end{align}   \qquad\qquad(diml.2.5.6)
$$

可分解的拉格朗日函数：

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

前面提到，对偶提升方法求解优化问题时，目标函数必须满足强凸函数的条件，限制过于严格。为了增加对偶天提升法的鲁棒性和放松目标函数\\(f\\)的强凸约束条件，人们提出了[增广拉格朗日乘子法](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method)（Augmented Lagrangians）用于解决这类问题。

**为什么添加二次惩罚项就可以“解除对目标函数\\(f(x)\\)强凸性质”的限制了呢？**

回答这个问题我们需要从最初的等式优化问题入手，公式\\((diml.2.5.1)\\)的等式优化问题等价于下面的优化问题：

$$
\begin{align}
& \; \min_{x} \;\; f(x) + \frac{\rho}{2} {\Vert Ax-b \Vert}_2^2 \\\
& \quad s.t. \; Ax = b
\end{align} 	\qquad\qquad(diml.2.5.9)
$$

在约束条件\\(Ax-b=0\\)下，目标函数定义为\\(f(x) + \frac{\rho}{2} {\Vert Ax-b \Vert}_2^2\\) 与\\(f(x)\\)是等价的，因为最优解在满足约束条件的前提下优化目标的后一项等于0，这一项称为**二次惩罚项**（或惩罚函数项）。

那么，对应的拉格朗日表达式为：

$$
\mathcal{L}_{\rho}(x, \beta) = f(x) + \frac{\rho}{2} {\Vert Ax-b \Vert}_2^2 + \beta^T (Ax-b)    \qquad\quad(diml.2.5.10)
$$

下面对上述公式做一个解读：

> 
$$
\mathcal{L}_{\rho}(x, \beta) = \overbrace{f(x)}^{原优化目标} + \overbrace{\underbrace{\frac{\rho}{2} {\Vert Ax-b \Vert}_2^2}_{二次惩罚项} + \underbrace{\beta^T(Ax-b)}_{拉格朗日乘子项} }^{增广拉格朗日乘子项}
$$

Argumented Lagrangians方法的核心思想可以总结为：

| 通过引入惩罚函数项（二次项），放松对目标函数\\(f(x)\\)严格凸的限制，同时使得算法更加稳健。|
| --- |

到这里为止，我们只是在不改变原优化问题的前提下，添加了一个二次惩罚项，并没有回答**“为什么添加二次惩罚项就可以“解除对目标函数\\(f(x)\\)强凸性质”的限制了呢？”**这个问题。

**原优化目标\\(f(x)\\)如果是非凸的，那意味着拉格朗日函数在对\\(x\\)求偏导时 结果可能不可导，无法进行参数交替迭代优化。添加了二次项惩罚项之后，可以保证即便\\(f(x)\\)不满足强凸条件，在对x求偏导时 仍然可导，因为二次惩罚项部分可以保证这一点**。

因此，添加二次惩罚项的好处是可以保证增广的对偶函数一定是可微的。迭代公式如下：

$$
\begin{align}
x^{k+1} & := \arg \min_{x} \mathcal{L}_{\rho} (x, \beta^k) \qquad (step1) \\\
\beta^{k+1} & := \beta^k + \rho(Ax^{k+1}-b)  \quad\;\;(step2)
\end{align}  \qquad(diml.2.5.11)
$$

上式即为增广拉格朗日乘子法的交替迭代公式。与标准的对偶提升法类似，但有两点不同：

1. \\(step1\\). \\(\mathcal{L}(x, \beta)\\)对\\(x\\)求偏导得到最小值时用的是增广拉格朗日函数；
2. 惩罚项参数\\(\rho\\)在这里作为对偶变量更新时的步长（step size）。

**为什么选择\\(\rho\\)作为对偶变量更新的迭代的step size？**

这里假设\\(f\\)是可微的（尽管在Augmented拉格朗日法中不是必须的）。\\((x^{\star}, \beta^{\star})\\)是优化问题的最优解。此时应该满足如下条件：

$$
Ax^{\star} - b = 0,\; \nabla f(x^{\star}) + A^T\beta^{\star} = 0 \qquad (diml.2.5.12)
$$

根据参数交替迭代公式的\\(step1\\)，第\\(k+1\\)的迭代结果\\(x^{k+1}\\)是由最小化函数\\(\mathcal{L}_{\rho}(x, \beta^k)\\)得来（具体做法：对参数\\(x\\)求偏导，令偏导数等于0），所以\\(x^{k+1}\\)满足下式成立

$$
\begin{align}
0 & = \nabla_{x} \mathcal{L}_{\rho}(x^{k+1}, \beta^k) \\\
& = \nabla_{x} f(x^{k+1}) + \beta^k A + \rho( A^T Ax - A^T b) \\\
& = \nabla_{x}f(x^{k+1}) + A^T \left(\underline {\beta^k + \rho(Ax^{k+1} - b) } \right) \\\
& = \nabla_{x}f(x^{k+1}) + A^T \beta^{k+1}
\end{align} \qquad(diml.2.5.13)
$$

因此可以得到，\\(\rho\\)为对偶变量参数更新的步长，并得到了第\\(k+1\\)次迭代的最优解\\((x^{k+1}, \beta^{k+1})\\)。随着迭代的进行，原函数的残差\\(Ax^{k+1}-b\\)逐步收敛到0，得到最优解。

相比对偶提升法，增广拉格朗日乘子法有更好的收敛性质，并且拥有对目标\\(f(x)\\)不做强凸条件的限制，但是这些好处总是要付出一定的代价的：**如果目标函数是\\(f(x)\\)是可分的，此时增广拉格朗日函数\\(\mathcal{L}_{\rho}(x, \beta)\\)是不可分的（因为惩罚函数项部分涉及到矩阵相乘计算，无法用分块形式进行并行化求解），因此公式\\((diml.2.5.11)\\)的\\(step1\\)没有办法在分布式环境下并行优化**。

---

> **回答问题：为什么在优化领域，一般要求目标函数是凸的呢？**
>
> 参考：http://blog.csdn.net/yhdzw/article/details/39288581

---

如果能结合Dual Ascent的并行求解的优势 和 Augmented Lagrangians Methods of Multipliers鲁棒性和不错的收敛性质，那么就可以使大多数目标函数求解都能在分布式环境下并行实现，岂不美哉！！！

果然在2010年由Stephen Boyd大师等人系统性的整理了交替方向乘子法（ADMM算法），可以解决上述的问题。为此它们长篇论述了ADMM算法的演化过程，参考论文：[Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers](http://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf) 该文被《Foundations and Trends in Machine Learning》录入。

> 从文章题目上可以看出ADMM算法的适用场景：分布式优化 && 统计学习。
> 
> 分布式优化很好理解，因为数据量比较大以至于需要用多台机器来求解优化问题；统计学习就是能够通过计算充分统计量得到最优解的学习问题。 那么，**ADMM算法适用于大规模统计学习在分布式环境下的优化求解问题**。
> 
> 说明：本章ADMM论述，均源于此文和对此文的理解。

### 交替方向乘子法（ADMM）

交替方向乘子法（Alternating Direction Method of Multipliers，简称ADMM）可以理解为增广拉格朗日乘子法的变种。旨在整合对偶提升法的可分解性和增广拉格朗日乘子法优秀的收敛性质，进一步提出的新算法。

我们修改下公式\\((diml.2.5.1)\\)，这样更符合统计学习目标函数的形式. 重新定义的优化问题：

$$
\begin{align}
& \min \quad f(x) + g(z) \\\
& s.b. \quad Ax + Bz = C
\end{align}  \qquad\qquad(diml.2.5.14)
$$

增广拉格朗日函数

$$
\mathcal{L}_{\rho}(x,z,\beta) = f(x) + g(z) + \underline{ \frac{\rho}{2} {\Vert Ax+Bz-C \Vert}_2^2 + \beta^T(Ax+Bz-C) } \qquad(diml.2.5.15)
$$

其中\\(x \in R^n, z \in R^m; A \in R^{p \times n}, B \in R^{p \times m}, C \in R^p\\)。按照乘子法的思路，参数交替更新迭代公式为：

$$
\begin{align}
(x^{k+1}, z^{k+1}) & := \arg \min_{x,z} \mathcal{L}_{\rho}(x,z,\beta^k) \qquad\qquad(step1) \\\
\beta^{k+1} & := \beta^k + \rho (Ax^{k+1} + Bz^{k+1} - C) \quad(step2)
\end{align}  \qquad(diml.2.5.16)
$$

上式中的\\(step1\\)要求对两个原始联合最小化。我们看 ADMM算法的参数更新方式：

$$
\begin{align}
x^{k+1} & := \arg \min_x \mathcal{L}_{\rho}(x, z^{k}, \beta^k) \qquad\qquad (step1) \\\
z^{k+1} & := \arg \min_z \mathcal{L}_{\rho}(x^{k+1}, z, \beta^k) \qquad\quad\; (step2) \\\ 
\beta^{k+1} & := \beta^{k} + \rho(Ax^{k+1} + Bz^{k+1} - C) \quad\;\; (step3)
\end{align}  \qquad\qquad(diml.2.5.17)
$$

不同于公式\\((diml.2.5.16)\\)，ADMM算法采用交替方式迭代（又称序贯式迭代），称为**交替方向**(alternating direction)。拆分参数\\(x\\)和\\(z\\)两步迭代最大的好处是：**当\\(f\\)和\\(g\\)都可分时，目标**

从ADMM名字上可以看出，它Augmented Lagrangians上添加了交叉方向（Alternating Direction），然后通过交叉换方向来交替优化。ADMM优化算法框架的结构形式如下：

	

	
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

+ [分布式计算、统计学习与ADMM算法](http://joegaotao.github.io/cn/2014/02/admm/)

### PPT

+ Precusors of ADMM
	+ 受限约束优化
		+ 原问题，拉格朗日，对偶问题（以SVM和最大熵模型为例），强对偶性以及成立条件 
	+ 对偶提升法
		+ 本质上是用梯度提升，求两个参数（x, beta），给出迭代公式；
		+ 之所以称为对偶提升，随着迭代进行，对偶函数是递增的。
		+ 梯度提升可用于对偶函数g不可导的场景，使用次梯度。
	+ 

