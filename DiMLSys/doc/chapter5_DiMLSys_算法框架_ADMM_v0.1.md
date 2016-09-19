## chapter4: DiMLSys之算法框架-ADMM

+ author: zhouyongsdzh@foxmail.com
+ date: 20160110

### 写在前面

分布式机器学习系统，一定要有链接硬件的软组织－分布式算法框架。这里要介绍的分布式算法框架是ADMM算法（Alternating Direction Multiplier Method，交叉方向乘子法）。

ADMM算法结构天然地适用于分布式环境下具体任务的求解。

### 约束优化问题一般解决方案

#### 对偶上升法（Dual Ascent）

对于一个有约束条件的优化问题来说，通常的求解方式是构造拉格朗日函数，然后利用**交替优化**的思路，不断地迭代参数和对偶变量（乘子）参数updated表达式，使得两者同时达到optimal。这就是对偶上升法（Dual Ascent）求解的核心思想。

> 1 给出公式表达式；
> 


对偶上升法有一个优点就是**如果目标函数是可分的，它可以在分布式环境下求解**。但是它也有一个很大的缺点就是**目标函数必须是强凸函数**。这个对于一般的凸函数，如果用对偶上升法求解参数，并不能得到最优解。




### ADMM算法框架


$$
xx
$$

### ADMM算法应用

| 算法框架 | 模型 | 参数学习方法 |
| --- | :--- | --- | 
| admm | Lasso <br> Logistic Regression <br> Factorization Machine <br> Filed-awared Factorization Machine | sgd <br> adaptive sgd <br> ftrl <br> lbfgs |

#### ADMM + Lasso

#### ADMM + Logistic Regression

#### ADMM + Factorization Machine

#### ADMM + Filed-awared Factorization Machine