title: "第03章：DiML－系统架构" 
mathjax: true
date: 2015-01-05 18:30:31
categories: 
	- 分布式机器学习
tags: 
	- ADMM
	- MPI
	- Rabit
---

+ author: zhouyongsdzh@foxmail.com
+ date: 2016-03-29
+ weibo: [@周永_52ML](http://weibo.com/p/1005051707438033/home?)


### 目录

+ [1. 系统架构](#1.机器通信概述)
+ [2. 同步通信-MPI](#2.同步通信-MPI) 
+ [3. 同步通信-Rabit](#3.同步通信-Rabit)
    + [3.1. 功能](#3.1.功能)
    + [3.2. 容错机制](#3.2.容错机制) 
    + [3.3. 应用示例](#3.3.应用示例)
+ [4. 异步通信-Parameter Server](#4.异步通信-Parameter-Server)
    + [4.1. 工作原理](#4.1.工作原理)
    + [4.2. 应用场景](#4.2.应用场景) 
+ [5. OpenMIT与机器通信](#5.OpenMIT与机器通信)
    + [5.1. MPI、Rabit和PS比较](#5.1.MPI-Rabit和PS比较) 
    + [5.2. OpenMIT与机器通信](#5.2.OpenMIT与机器通信)


<h3 id="3.同步通信-Rabit">3. 同步通信-Rabit</h3>
---

<h4 id="3.1.功能">3.1. 功能</h4>
---

<h4 id="3.2.容错机制">3.2. 容错机制</h4>
---

Rabit相比于MPI除了轻量化、与其它系统结合实现可插拔的优点外，还有一个非常大的优势就是容错机制（Fault Tolerance）。Rabit提供的容错机制可以保证任务在执行过程中，不会因为某个计算节点故障导致整个任务执行失败。它通过checkpoint保存的信息可以快速“修复”新分配的计算节点，从而保证任务继续正常运行。

下面我们通过图示来看rabit是怎么解决容错问题的（该图来自[rabit github]()）

![]()

图示的场景描述如下：

1. 节点1在第二次checkpoint之后Allreduce之前执行失败（故障等原因）；
2. 此时其它的节点在第二次执行Allreduce前停下来等待，为了是帮助节点1恢复状态；
3. 当节点1重启时（重新分配计算资源，比如yarn），它会掉用`LoadCheckPoint`方法获得其它任意一个存在的节点最近的checkpoint；
4. 这是节点1从最近的一个checkpoint处开始继续执行；
5. 当节点1再次掉用第一个Allreduce时，因为其它节点已经知道了结果，节点1可以从它们中的任意一个得到；
6. 当节点1到达第二个Allreduce时，其它的节点发现节点1已经追赶上来（catched up，表示节点1已经修复完成）. 这是它们一起继续正常运行。

这个容错机制基于Allreduce和Broadcast的一个关键属性：**在执行完`Allreduce/Broadcast`之后所有的节点将获得相同的结果**。因为这个属性，任何节点将记录历史掉用`Allreduce/Broadcast`之后的结果。当一个节点需要修复时，它将从一些“或者的”节点获得丢失的结果并且重构自身。

**checkpoint是什么？** checkpoint作为“检查点”会保存最近一次掉用`Allreduce/Broadcast`后的结果。它保存在内存中主要用于备份。每个计算节点的checkpoint是由用户定义的，并被分为两部分：**全局模型和局部模型**。全局模型由所有节点共享被所有节点备份。The local model of a node is replicated to some other nodes(selected using a ring replication strategy). 

Rabit的容错过程不同于`失败-重启`策略（任一节点失败后，所有的节点在同一个checkpoint处重启，MPI采用该机制）。在rabit中，所有“活着的”节点在执行Allreduce时被阻塞，并且帮助“失败节点”恢复。为了追赶大部队，被恢复的节点获取它的最近的checkpoint和`Allreduce/Broadcast`调用结果。

上述仅仅是对rabit容错机制的概念性介绍。实现时会更复杂，因为需要处理更复杂的场景，比如多个节点同时失败、在节点恢复节点再次失败等。

<h4 id="3.3.应用示例">3.3. 应用示例</h4>
---

+ 单机模拟多个workers执行rabit程序

```
cd ${rabit_dir}/guide && make   // 生成basic.rabit
../tracker/rabit_demo.py -n 2 basic.rabit
```

<h3 id="4.异步通信-Parameter-Server">4. 异步通信-Parameter Server</h3>
--

随着数据量和参数的快速增长，单机搞不定，需要利用集群分布式方式搞定。


#### 1. PS系统架构

图

PS架构主要包含两类节点：Clients和Servers。

2. Servers维护全局共向参数，同步参数和全局参数更新；其中每个Server只维护全局参数的一部分；
1. Clients执行在数据上的计算任务，每个client只获取部分数据，并计算局部统计量 比如Gradient；


#### 2. 接口

1. 数据接口：Key-Value Vectors
2. 通信接口：Pull && Push
3. 用户自定义函数：

### 3. 一致性问题 与 系统容错



1. PS系统架构
    + 架构图
    + 解释
2. PS特点
    + 异步：提高效率；
    + 通信开销：Snappy压缩；vector clock节省空间；
    + 宽松一致性：
    + ...

<h4 id="4.2.应用场景">4.2. 应用场景</h4>
---

PS非常适合解决分布式优化问题中的参数通信问题，假设我们要优化的问题如下：

$$
\min_w \quad \sum_{i=1}^m \; f(x^{(i)}, y^{(i)})
$$

其中，\\((x^{(i)}, y^{(i)})\\)表示样本特征和label，\\(w\\)是参数权重。我们考虑用mini-batch SGD方法求解参数，每个batch样本数为b个。在\\(t\\)时刻，算法首先随机挑选b个样本，利用下面的公式更新权重\\(w\\)：

$$
w \longleftarrow w - \eta_t \sum_{i=1}^b \nabla f(x^{k_i}, y^{k_i}, w)
$$

下面通过两个示例来说明ps-lite是如何完成分布式优化算法的。

+ 异步SGD

首先看异步SGD是如何工作的？我们让servers的个数保持w个，其中第k个server保存参数w的一部分，用\\(w_{k<|sub>}\\)表示。一旦从worker接受梯度值，server k更新该部分参数：

```
t = 0;
while (Received(&grad)) {
  w_k -= eta(t) * grad;
  t++;
}
```

如果接受的梯度来自任意的worker节点，那么函数`received`将返回true???。\\(eta\\)将返回在t时刻的学习率。

然而对于worker节点，在每个时刻它将做下面4件事：

```
Read(&X, &Y);       // read a minibatch X and Y
Pull(&w);           // pull the recent weight from the servers
ComputeGrad(X, Y, w, &grad);    // compute the gradient
Push(grad);         // push the gradients to the servers
```

ps-lite在这个过程中将发挥什么作用呢？ps-lite将提供`push`和`pull`功能，这两个函数将在servers之间通信获取对应的参数数据。

> 注意：

+ 同步SGD
