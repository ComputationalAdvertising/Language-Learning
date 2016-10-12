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

参数迭代步骤为：

$$
\begin{align}
w_t^{k+1}  \longleftarrow & \arg\min \; \frac{1}{m_t}\; L(\mathbf{y}_t, \mathbf{x}_t w_t) + \beta_t^T w_t + \frac{\rho}{2} {\Vert w_t - \theta^k \Vert}_2^2 \qquad\qquad\;\;(1) \\\
&  \arg\min \; \frac{1}{m_t}\; L(\mathbf{y}_t, \mathbf{x}_t w_t) + \frac{\rho}{2} {\Vert w_t - \theta^k + \frac{1}{\rho} \beta_t^k \Vert}_2^2 \\\
\theta^{k+1} \longleftarrow & \arg\min \; \lambda {\Vert \theta \Vert}_1 - \sum_{t=1}^{T} (\beta_t^k)^T \theta + \frac{\rho}{2} \sum_{t=1}^{T} {\Vert w_t^{k+1} - \theta \Vert}_2^2 \qquad\qquad(2) \\\
\beta_t^{k+1} \longleftarrow & \beta_t^k + \rho(w_t^{k+1} - \theta^{k+1}) \qquad\qquad\qquad\qquad\qquad\qquad\qquad\;\,(3)
\end{align} 
$$


