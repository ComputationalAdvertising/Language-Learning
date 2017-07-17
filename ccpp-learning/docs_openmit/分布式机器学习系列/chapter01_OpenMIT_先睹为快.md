## DiML-综述

+ Author: BaiGang, ZhouYong
+ Date: 2015.12.10～

参考知乎@白刚的回答：https://www.zhihu.com/question/37057945

机器学习研究的是从**数据**中自动归纳逻辑或规则，并根据归纳结果对新数据进行预测的**算法**。这个方向是与数据相关的，而从前数据的收集过程效率很低且成本很高，很少有大规模的问题需要解决。直到互联网发展起来，Web服务器记录和收集大量的用户浏览、点击、下单、支付、交互、分享等行为信息，这些数据的特点：收集成本低、数据量大且有价值（数据规律）。由此，“大规模机器学习”变得可行，并且越来越重要。

### 机器学习－模型问题
---

在机器学习的应用场景上，我们接触到的各种问题、算法、技术看似复杂，但主要可以归纳为两个方面：

+ 根据要建模的问题场景和数据规模，确定模型的**representation**方案
+ 在representation的无限可能中去寻找最优模型的**optimization**方法

“大规模机器学习”所涉及的就是从模型的representation和optimization这两个方面去解决**应用大规模数据**时的**理论和工程**上的问题。

### 机器学习－学习过程
---

从模型的representation和optimization这两个方面，形式化地定义机器学习的求解问题，learning的过程大概可以这么描述：

$$
\min_\vec{w} \; \sum_{(x, y) \in \mathcal{D}} \mathcal{L}(x,y; \vec{w}) \; + \; \Omega(\vec{w})
$$

其中，$\vec{w}$就是representation，或者说是模型参数；$\mathcal{D}$是数据；$\mathcal{L}(\bullet)$和$\Omega(\bullet)$分别表示loss/risk/objective和prior/regularization/structural knowledge.

$min_{\vec{w}}$的过程即是前面说的optimization，通常是迭代优化的过程。有些可以转化为最优化问题，通过数值优化的方法，如batch的GD/LBFGS，或者onlineSGD／FTRL等进行参数估计；有些无法构造成可解的最优化问题时，转化为概率分布的估计问题，通过probabilistic inference来解决，比如用Gibbs Sampling来训练Latent Dirichlet Allocation模型。

无论是数据优化，还是Sampling，都是不断迭代的过程：

$
\{ \\\
\quad \text{输入}：\vec{w}^0 = \text{init model}；训练数据集 \mathcal{D} \\\
\quad \text{输出}: 优化后的\vec{w} \\\
\quad 训练过程 \\\
\quad \text{for i} \leftarrow \text{1 to T} \qquad // 迭代 \\\
\quad \text{do} \\\
\qquad \nabla^i = \text{gradient}(\mathcal{D}, \vec{w}^{(i-1)}) \qquad\; \text{(1)} \\\
\qquad \vec{w}^i = \vec{w}^{(i-1)}  + \mu(\nabla^i) \qquad\qquad \text{(2)} \\\
\quad \text{done} \\\
\}
$

每一步迭代做两件事：1. 当前的模型$\vec{w}^{i-1}$在数据集$\mathcal{D}$上的evaluation，得到一个“与最好模型相比可能存在的偏差”量；2. 根据这个偏差量，去修正模型的过程。

“大规模机器学习”就是要解决数据$\mathcal{D}$和模型$\vec{w}$的规模非常大时涉及到的**理论和工程**上的问题。

### 大规模机器学习－理论基础
---

我们先分析一下大规模机器学习理论上的问题，即**为什么“大规模”是有益的，比“小规模”优势在哪里？**

模型训练的目标是使模型应用到新数据上效果最好，即模型的泛化能力要好，表现在generalization error最小。对于监督学习，generalization error理论上有两个来源：**bias和variance**。

+ high bias : 可以看作模型的建模能力不足，在训练集上的误差较大，并且在新数据集上往往更差，也就是under-fitting;
+ high variance : 可以看做模型过于拟合训练集，而对新数据效果很差，也就是over-fitting.

所以对于模型效果上，除了特征工程及其trick外，“调得一手好参”－解决好bias和variance的tradeoff－是算法工程师的核心竞争力之一。但“大规模机器学习”可以在一定程度上同时解决上面两个问题：

+ **解决high bias／under-fitting : 提升模型复杂度，提高模型表达能力。** 通过引入更多的特征（二阶、三阶组合特征等）、更复杂的模型结构，让模型可以更全面的描述数据的概率分布／分界面／逻辑规则，从而有更好的效果；
+ **解决high variance／over-fitting : 增大训练样本集。** variance的引入可以理解为样本集不够全面，不足以描述数据规律，即训练样本的分布与实际数据的分布不一致造成的。扩大样本集，甚至使用全量数据，可以尽量使得训练集与应用模型数据集的分布一致，减少variance的影响。

### 大规模机器学习－计算架构
---

“知识体系”可以从两个方面来构建：一方面是宏观上的架构；另一方面是无数微观上的trick。学习过程中的主要的两步迭代决定了宏观上的架构应该是什么样子。两个步骤内的计算逻辑和之间的数据交互方式引入了解决这个特定问题的技术和trick。

首先，$\mathcal{D}$和$\vec{w}$需要分布在多个计算节点上。

假设数据分布在n个数据节点上，模型分布在m个参数节点上。数据集仅与第一步有关，这部分计算应该在数据所在的节点上进行；第二步是对模型的更新，这一步应该在模型所在的参数节点上进行。也就是说，这个架构中有两个主要的角色，每个角色都有自己的计算逻辑。并且在分布式系统中，需要有一套replica的机制来容错。

几乎所有的分布式机器学习平台的系统都可以看做主要由这两个角色构成。

在Spark MLLib里，driver program是model node，executor所在的worker node是data node；在Vowpal Wabbit里面，mapper id为0的节点扮演了model node的角色，所有的mapper扮演了data node的角色；在Parameter Server里面，server是model node，worker是data node。其中MLLib和VW的model node都是单节点，而PS可以把模型扩展到多个节点。

其次，节点之间需要传输数据。data node需要在计算前获取当前的模型$\vec{w}^{i-1}$，model node需要得到更新量$\nabla^{i}$.

MLLib通过RDD的treeAggregate接口，VW通过allReduce，这两者其实是相同的方式。节点被分配在一个树状的结构中，每个节点计算好自己的这部分更新量$\nabla_{k}^{i}$后，汇总它的子节点结果，传给它的父节点。最终，根节点（model node）获取到总的更新量$\nabla^{i} = \sum_k \nabla_k^i$。DMLC中的同步通信库[Rabit]()对这类aggregation数据传输提供了更简洁的接口和更鲁棒的实现。我们用Rabit实现了一套multi-task learning去训练不同广告位上广告点击率模型的方案。

而Parameter Server则是通过Worker去Pull其需要的那部分$\vec{w}^{i-1}$，计算更新量并把自己的那部分$\nabla_k^i$去Push给Server，Server去做对应的模型更新。这个似乎比前面的aggregation要简单，其实不然，当每个Worker进度差异特别大时（数据加载／网络通信／机器资源等原因导致），又引入了新的问题。

第三，分布式系统的并行和一致性问题。

分布式系统中每个节点的状态都不相同，计算的进度也会不一样。当某个data node计算的更新量$\nabla_k^i$对应的是若干轮迭代之前的$\vec{w}^{i-n}$时，它对优化不一定有贡献，甚至可能影响到收敛。

MLLib和VW上的迭代计算，可以看作是**同步**的。模型节点获取了所有数据节点的$\nabla_k^i$汇总起来，再更新$\vec{w}^{i}$，并BroadCast给所有的数据节点。每一轮迭代（Epoch）模型节点获取的是完整的更新量，数据节点拿到的是一致的模型。所以这里相当于设置了barrier, 等所有节点都到达这一轮的终点时，再放所有数据节点开启下一轮计算。

Paremeter Server上的迭代计算，如果完全不设置barrier，也即是Asynchronous Processing，会让整个计算过程（梯度与参数更新）尽快的推进下去，但数据节点之间的**进展程度差异太大会造成优化求解的收敛性不稳定**；如果每一轮迭代都设置barrier，也就是Bulk Synchronous Processing，会保证数据的一致，但是会出现在每一轮快节点等待慢节点，造成计算效率低下的问题。

所以一个折衷的解决方案就是设置Barrier。不同于同步每一轮都设置barrier，而是限定一个时间窗口长度$\delta$。计算最快的节点也要等到所有的$\nabla_{k}^{i-\delta}$都被应用到$\vec{w}^{i-\delta}$的更新之后再开启下一轮“Pull&Compute&Push”。这种机制又叫Bounded Delay Asynchronous或者Stale Synchronous Processing.

综述以上，大规模机器学习的分布式计算框架特点总结如下：

| 分布式计算框架 | 模型节点 | 数据节点 | 模型节点个数 | 通信接口 | 通信方式 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| Spark MLLib | Driver Program | Executor | 1 | treeAggregate | 同步 |
| Vowpal Wabbit／MPI | Mapper (id=0) | All Mapper | 1 | allReduce | 同步 |
| **Parameter Server** | Server | Worker | **多个** | Push/Pull | **异步** |

总的来说，解决好以上几个问题、构建一套大规模的机器学习系统并非不可实现，但是它需要严谨的工程能力和大量巧妙的trick。工程能力可以帮助更合理地抽象结构、定义同用接口，trick则帮助解决现实中的障碍，得到更稳定和高效的系统。

### 大规模机器学习－系统构建
---

机器学习任务属于**密集型纯占用CPU的数值计算**任务，所有计算需要的存储资源都在任务启动时申请，训练过程基本不在申请内存。并且分布式环境下需要较高的网络通信性能。

所以大规模机器学习的系统或平台应该是具有这样的特性：

+ 较高的CPU、内存性能和网络通信带宽；
+ 较灵活的模块抽象与接口；
+ 能够与现有的大数据生态系统、平台兼容。

综合以上节点，现有的若干种针对大规模机器学习的开源工具种，DMLC（分布式机器学习社区）是值得使用的一个。其特点主要有：

+ 性能：C++实现，引入大量的C++11特性来提升效率和代码可读性；
+ 模块抽象与接口：
    + rabit：实现了可容错的allreduce接口；
    + ps-lite：实验了parameter server，并包含大量工程上的优化和trick；
+ 兼容大数据生态系统：由dmlc-core来支持
    + Yarn资源管理：可以直接在现有的Hadoop 2.0+／Spark集群上运行；
    + HDFS/S3等分布式文件系统接口，可以直接读取现有的Hadoop集群上的资源，与ETL过程无缝结合。

### SpaceX: 基于Parameter Server的分布式机器学习工具包

+ 1. SpaceX机器学习
    1. Worker端&&Server端执行逻辑（为代码）［doing］

### Worker端工作过程：

$
\{ \\\
\quad \text{Input}：\text{train: } \mathcal{D}；\text{valid: } \mathcal{V}；\text{conf}； \\\
\quad \text{Processing}：\\\
\qquad 01. \;\text{initialize}: model, loss, \mathcal{D_t, V_t}； \\\
\qquad 02. \;\text{for } epoch \leftarrow \text{1 to K}  \qquad\qquad \text{// mex epoch} \\\ 
\qquad 03. \;\text{do} \\\
\qquad 04. \quad \text{for } batch.data \text{ in } \mathcal{D_t} \qquad\quad \text{// mini-batch } \\\
\qquad 05. \quad \text{do} \\\
\qquad 06. \qquad \vec{w}_{l} = Pull(batch.data)；\;\; \text{// from Servers} \\\
\qquad 07. \qquad \nabla_l =  Gradient(batch.data, \vec{w}_l)； \\\ 
\qquad 08. \qquad Push(\nabla_l)； \qquad\qquad\quad\; \text{// to Servers} \\\
\qquad 09. \quad \text{done} \\\
\qquad 10. \quad \text{metric} = Metric(\mathcal{D_t, V_t})；\\\
\qquad 11. \quad  \text{Send metric to Scheduler}；  \\\
\qquad 12. \;\text{done} \\\
\qquad 13. \;\text{Send signal "worker.done" to Server. } \\\
\qquad 14. \;\text{finish.} \\\
\}
$

### Server端工作过程：

$
\{ \\\
\quad \text{Input: conf；} \\\  
\quad \text{Output: } \vec{w} \qquad \text{// optimized model} \\\
\quad \text{Processing: } \\\
\qquad 01. \;\text{initialize: } optimizer, \vec{w}, \cdots etc；   \\\
\qquad 02. \; \text{while(true)} \\\
\qquad 03. \quad req = Recieve(\text{info})； \\\
\qquad 04. \quad \text{if req is "push"; then} \\\
\qquad 05. \qquad \nabla_l = req.gradient \\\
\qquad 06. \qquad \text{if is.async; then} \\\
\qquad 07. \qquad\quad AsyncUpdate(\nabla_l)； \\\ 
\qquad 08. \qquad \text{else：} \\\
\qquad 09. \qquad\quad SyncUpdate(\nabla_l)； \\\
\qquad 10. \qquad \text{endif} \\\
\qquad 11. \quad \text{else if req is "pull"; then} \\\
\qquad 12. \qquad \vec{w_l} = Collect(req.keys)；  \\\
\qquad 13. \qquad \text{Send } \vec{w_l} \text{ to Worker}；\\\
\qquad 14. \quad \text{else if req is "signal.xxx"; then} \\\
\qquad 15. \qquad \text{do some thing.} \\\
\qquad 16. \quad \text{else if req is "worker.done"; then} \\\
\qquad 17. \qquad SaveModel(\vec{w})； \\\
\qquad 18. \qquad \text{Send signal "server.done" to Scheduler; } \\\
\qquad 19. \qquad break；\\\
\qquad 20. \;\text{finish.} \\\
\}
$

Scheduler端功能

+ 

$
\{ \\\
// todo \\\
\}
$

2. 计算引擎（Worker&&Server）[todo]

        1. 画图

3. 模型、优化算法、损失函数、评估指标；

+ 模型

| 模型 | 表达式 | 特点 |
| :--: | --- | --- |
| Logistic Regression | $\hat{y}(\mathbf{x}) = w_0 +  \sum_{i=1}^{n} w_i x_i;\quad sigmoid$ | 线性表达
| Factorization Machine | $\hat{y}(\mathbf{x}) := w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$ | 1. FM降低了交叉项参数学习不充分的影响<br>2. FM提升了参数学习效率<br>3. FM提升了模型预估能力(历史未出现场景) 
| Field-awared Factorization Machine | $\hat{y}(\mathbf{x}) := w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \mathbf{v}_{i,\,f_j}, \mathbf{v}_{j,\,f_i} \rangle x_i x_j$ | FM基础上，学习每个field的隐式表达（稀疏性增大，对训练数据要求更多） |
| ... | ... | ... |

后续可以支持Large-Scale Matrix Factorization等模型.

+ 优化算法

| 优化算法 | 参数更新 | 主要特点 |
| :--: | --- | --- |
| GD | $w_i \leftarrow w_i - \eta \cdot g_i$ | 全局学习率 | 
| AdaGrad | $w_i \leftarrow w_i - \frac{\eta}{\sqrt{n_i} + \epsilon} \cdot g_i; \quad n_i \leftarrow n_i + g_i \cdot g_i;$ | 1. 每个参数有自己的学习率，初始化后动态调整; <br> 2. 缺点：参数更新能力越来越弱 | 
| AdaDelta | $w_i \leftarrow w_i - - \dfrac{RMS[\Delta w]_{t-1}}{RMS[g]_{t}} \cdot g_i;$ <br> $E[\Delta w^2]_t = \gamma E[\Delta w^2]_{t-1} + (1 - \gamma) \Delta w^2_t; $ |  1. 改进：梯度平方和改为历史梯度平方的衰减平均值；<br>2. 自适应学习率<br>
| FTRL | $z_i \leftarrow z_i - sigma \cdot w_i; \quad n_i \leftarrow n_i + g \cdot g;$ <br> $w_i \leftarrow 0, \quad \text{if } \vert {z_i} \vert \le \lambda_1;$ <br> $w_i \leftarrow  \left( \frac{\beta +\sqrt{n_i} } {\alpha} + \lambda_2 \right)^{-1} \cdot \left(\lambda_1 \cdot sign - z_i \right) \quad othersize.$ | 1. 学习精度和模型稀疏性二者兼备 <br> 2.  Per-Coordinate Learning Rate
| ... | ... |

后续可以支持Limited-BGFS, Alternating Least Squares, Markov Chain Monte Carlo等参数学习算法。

+ 损失函数

| 损失函数 | 表达式 | 适用模型 | 
| :--: | --- | --- |
| **平方损失**<br>(Squared Loss Error) | $\frac{1}{2} (y^{(i)} - f(x^{(i)}))^2$ | Linear Model, FM, FFM, L2Boosting等 |
| **对数损失**<br>(Log Loss) | $\log (1+e^{- y^{(i)} \cdot f(x^{(i)})})$ | LR, FM, FFM, LogitBoost等 |

后续还可以支持 绝对损失(Absolute Error) ：$\vert y^{(i)} - f(x^{(i)}) \vert$，指数损失(Exponentail Loss)：$\exp(- y^{(i)} \cdot f(x^{(i)}))$..

+ 效果评估
    + 支持AUC，LogLoss，ACC评估方法 


    4. 性能压测
        1. 柱状图
    5. 如何使用
        1. 脚本与conf文件
    6. 后续计划


### 大规模机器学习与OpenMIT

OpenMIT是基于分布式系统实现的机器学习工具包，同时拥有大规模机器学习工具的MPI和Parameter Server的实现。不同于Parameter Server的全局模型更新的分布式优化算法，基于MPI实现大规模机器学习用到的分布式算法框架是ADMM算法，它更适用于multi-task learning任务。

+ 基于MPI的大规模机器学习
+ 基于Parameter Server的大规模机器学习
