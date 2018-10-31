title: "第6章：DiMLSys-算法模型-学习器" 
mathjax: true
date: 2015-06-01 18:30:31
categories: 
	- 分布式机器学习
tags: 
	- 通用线性模型
	- 通用加法模型
	- FM
	- FFM
	- Matrix Factorization
	- ADMM
	- 指数族分布
	- 多任务学习
	- Multi-Task Learning
---

+ author: zhouyongsdzh@foxmail.com
+ date: 2016-06-01
+ weibo: [@周永_52ML](http://weibo.com/p/1005051707438033/home?)
---

## ADMM参考资料：

+ 分布式计算、统计学习与ADMM算法：http://joegaotao.github.io/cn/2014/02/admm/
+ 关于ADMM的研究（二）: http://www.mamicode.com/info-detail-908450.html
+ 

**目录**

+ Multi-Task Learning
+ ADMM + Lasso
+ ADMM + Group Lasso
+ ADMM + LR


### Multi-Task Learning

优化目标：

$$
\min_{w} \quad \frac{1}{m} \sum_{i=1}^{m} L(y^{(i)}, w^T x^{(i)})
$$

\\(L(y, w^Tx)\\)为模型损失函数。

优化问题拆解为多任务形式：

$$
\begin{align}
\min_{w_0, w_1, \cdots, w_{T}} \quad \sum_{t=1}^{T} \left(\frac{1}{m_t} \sum_{i=1}^{m_t} L(y_t^{(i)}, w_t^Tx_{t}^{(i)})\right)
\end{align}
$$

数据集切分为\\(T\\)块，每一块作为一个子任务进行参数学习。

### ADMM + Lasso

L1正则化的线性回归模型。\\(L(y^{(i)}, w^Tx^{(i)}) = \left(y^{(i)} - w^Tx^{(i)} \right)^2\\)优化目标：

$$
\min_{w} \quad \frac{1}{m} \sum_{i=1}^{m} \left(y^{(i)} - w^Tx^{(i)} \right)^2 + \lambda {\Vert w \Vert}_1
$$

用ADMM形式改写：

$$
\begin{align}
& \min_{w_1, w_2, \cdots, w_T} \quad \sum_{t=1}^{T} \left( \frac{1}{m_t} \sum_{i=1}^{m_t} \left(y_t^{(i)} - w_t^T x_t^{(i)} \right)^2 + \frac{\lambda}{T}{\Vert \theta \Vert}_1 \right) \\\
& \quad\; s.b \qquad\; w_t=\theta \;(t=1,\cdots,T)
\end{align}
$$

令\\(\frac{1}{m_t} \sum_{i=1}^{m_t} \left(y_t^{(i)} - w_t^T x_t^{(i)} \right)^2 = L(\mathbf{y}_t, \mathbf{x}_t w_t), \; \\)，增广拉格朗日函数：

$$
\mathcal{L}(w_t, \theta, \beta_t) = \sum_{t=1}^{T} \frac{1}{m_t} \; L(\mathbf{y}_t, \mathbf{x}_t w_t) + \lambda {\Vert \theta \Vert}_1 + \sum_{t=1}^{T} \beta_t^T(w_t - \theta) + \frac{\rho}{2} \sum_{t=1}^{T} {\Vert w_t - \theta \Vert}_2^2
$$

对偶变量更新推导(20161106)：

定义参数\\(r_t = w_t - \theta, \mu_t = \frac{1}{\rho} \beta_t \\)，那么增广拉格朗日公式该写为：

$$
\sum_{t=1}^{T} \left(\rho \mu_t \cdot r_t + \frac{\rho}{2} {\Vert r_t \Vert}_2^2 \right) = \frac{\rho}{2} \sum_{t=1}^{T} \left({\Vert r_t + \mu_t \Vert}_2^2 - {\Vert \mu_t \Vert}_2^2 \right)
$$


$$
\frac{\partial \left(\frac{\rho}{2} \sum_{t=1}^{T} \left({\Vert r_t + \mu_t \Vert}_2^2 - {\Vert \mu_t \Vert}_2^2 \right) \right) }{\partial \mu_t} = \rho \cdot r_t = \rho \cdot (w_t - \theta)
$$

对偶变量的更新公式：



参数迭代步骤：

+ 局部参数的更新

$$
\begin{align}
w_t^{k+1}  \longleftarrow & \arg\min \; \frac{1}{m_t}\; L(\mathbf{y}_t, \mathbf{x}_t w_t) + \beta_t^T w_t + \frac{\rho}{2} {\Vert w_t - \theta^k \Vert}_2^2 \qquad\qquad\;\;(1) \\\
&  \arg\min \; \frac{1}{m_t}\; L(\mathbf{y}_t, \mathbf{x}_t w_t) + \frac{\rho}{2} {\Vert w_t - \theta^k + \frac{1}{\rho} \beta_t^k \Vert}_2^2
\end{align}
$$

+ 全局参数的更新

	$$
\theta^{k+1} \longleftarrow \arg\min \; \lambda {\Vert \theta \Vert}_1 - \sum_{t=1}^{T} (\beta_t^k)^T \theta + \frac{\rho}{2} \sum_{t=1}^{T} {\Vert w_t^{k+1} - \theta \Vert}_2^2 \qquad\qquad(2)
	$$

> 推导如下. 优化目标函数，对\\(\theta\\)求偏导：
>
$$
\begin{align}
& \frac{\partial \, \mathcal{L}(w_t, \theta, \beta_t)} {\partial{\theta}} ＝ \frac{\partial \; \left({\lambda {\vert \theta \vert}_1} - \sum_{t=1}^{T}(\beta_t)^T \theta + \frac{\rho}{2} \sum_{t=1}^{T} {\Vert w_t - \theta \Vert}_2^2 \right)} {\partial {\theta}} \\\
& \quad = \mathbf{sign}(\theta) \cdot \lambda - \sum_{t=1}^{T} \beta_t + {\rho} \sum_{t=1}^{T} \left(\theta - w_t \right) \\\
& \quad = \mathbf{sign}({\vert \theta \vert}_1) \cdot \frac{\lambda}{\rho} - \sum_{t=1}^{T} \left( \frac{\beta_t}{\rho} + w_t\right) + \sum_{t=1}^{T} \theta = 0 \\\
& 全局参数\theta更新公式整理得到：\mathbf{\underline{\theta = \frac{1}{T} \left(\sum_{t=1}^{T} 
\left( \frac{\beta_t}{\rho} + w_t\right) - sign({\vert \theta \vert}_1) \cdot \frac{\lambda}{\rho} \right) }}
\end{align}
$$
>
> 说明：全局参数的更新公式中，有l1-norm项需要求导。虽然其在0处不可导，但是仍有解析解，被称作软阈值（soft thresholding），也被称作压缩算子（shrinkage operator）。
>
$$
\theta =
\begin{cases}
\frac{1}{T} \left( \sum_{t=1}^{T} \left( \frac{\beta_t}{\rho} + w_t\right) + \frac{\lambda}{\rho}  \right) & \qquad \text{if} \; {\vert \theta \vert}_1 < 0, 若：\sum_{t=1}^{T} \left( \frac{\beta_t}{\rho} + w_t\right) + \frac{\lambda}{\rho} < 0. \\\
\frac{1}{T}\left(\sum_{t=1}^{T}\left(\frac{\beta}{\rho} + w_t \right) - \frac{\lambda}{\rho} \right) & \qquad \text{if} \; {\vert \theta \vert}_1 > 0, 若：\sum_{t=1}^{T} \left( \frac{\beta_t}{\rho} + w_t\right) - \frac{\lambda}{\rho} > 0. \\\
\; 0  & \qquad otherwise.
\end{cases}
$$
> 
> 全局参数的推导运用到了[软阈值（Soft-Thresholding）]()方法。
> 
> 参数：\\(\lambda\\) 为L1正则项系数；\\(\beta_t\\)对偶变量（拉格朗日乘子）; \\(\theta\\)全局参数; \\(w_t\\)局部参数;

+ 局部对偶变量的更新

$$
\begin{align}
\beta_t^{k+1} \longleftarrow & \beta_t^k + \rho(w_t^{k+1} - \theta^{k+1}) \qquad\qquad\qquad\qquad\qquad\qquad\qquad\;\,(3)
\end{align}  
$$

迭代公式说明：

+ 第1步：局部参数的更新。目标函数可以看作是```损失函数+L2正则项```(\\({\Vert w_t - const \Vert}_2^2\\))，局部参数更新涉及的参数有：\\((\theta, \beta_t, w_t)\\)；
+ 第2步：全局参数的更新。需要详细推到，涉及到软阈值. 全局参数的更新涉及参数：\\((w_1,\cdots,w_n, \rho, \beta_1,\cdots,\beta_n,\lambda)\\)
+ 第3步：局部对偶变量的更新。涉及参数：\\((\theta, \lambda, w_1,\cdots,w_n, \rho)\\)


---
## 资料


+ 论文
+ 相关资料博客
	+ [​DMLC对于机器学习和系统开发者意味着什么](http://weibo.com/p/1001603845546463886385?mod=zwenzhang)@陈天奇
