## DiML-综述

+ Author: BaiGang, ZhouYong
+ Date: 2015.12.10～

参考知乎@白刚的回答：https://www.zhihu.com/question/37057945

机器学习研究的是从**数据**中自动归纳逻辑或规则，并根据归纳结果对新数据进行预测的**算法**。这个方向是与数据相关的，而从前数据的收集过程效率很低且成本很高，很少有大规模的问题需要解决。直到互联网发展起来，Web服务器记录和收集大量的用户浏览、点击、下单、支付、交互、分享等行为信息，这些数据的特点：收集成本低、数据量大且有价值（数据规律）。由此，“大规模机器学习”变得可行，并且越来越重要。

在机器学习的应用场景上，我们接触到的各种问题、算法、技术看似复杂，但主要可以归纳为两个方面：

+ 根据要建模的问题场景和数据规模，确定模型的**representation**方案
+ 在representation的无限可能中去寻找最优模型的**optimization**方法

“大规模机器学习”所涉及的就是从模型的representation和optimization这两个方面去解决**应用大规模数据**时的**理论和工程**上的问题。

### 大规模机器学习-理论基础
---

在介绍大规模机器学习具体技术之前，我们先讨论一下**为什么“大规模”是有益的，比“小规模”优势在哪里？**

模型训练的目标是使模型应用到新数据上效果最好，即模型的泛化能力要好，表现在generalization error最小。对于监督学习，generalization error理论上有两个来源：**bias和variance**。

+ high bias : 可以看作模型的建模能力不足，在训练集上的误差较大，并且在新数据集上往往会更差，也就是under-fitting;
+ high variance : 可以看做模型过于拟合训练集，而对新数据效果很差，也就是over-fitting.

所以对于模型效果上，除了特征工程及其trick外，“调得一手好参”－解决好bias和variance的tradeoff－是算法工程师的核心竞争力之一。但“大规模机器学习”可以在一定程度上同时解决上面两个问题：

+ **解决high bias／under-fitting : 提升模型复杂性。** 通过引入更多的特征（二阶组合特征、三阶组合特征等）、更复杂的模型结构，让模型可以更全面的描述数据的概率分布／分界面／逻辑规则，从而有更好的效果；
+ **解决high variance／over-fitting : 增大训练样本集。** variance的引入可以理解为样本集不够全面，不足以描述数据规律，即训练样本的分布与实际数据的分布不一致造成的。扩大样本集，甚至使用全量数据，可以尽量使得训练集与应用模型数据集的分布一直，减少variance的影响。

### 机器学习-学习过程
---

从模型的representation和optimization这两个方面，形式化地定义机器学习的求解问题，learning的过程大概可以这么描述：

$$
\min_\vec{w} \; \sum_{(x, y) \in \mathcal{D}} \mathcal{L}(x,y; \vec{w}) \; + \; \Omega(\vec{w})
$$

其中，$\vec{w}$就是representation，或者说是模型参数；$\mathcal{D}$是数据；$\mathcal{L}(\bullet)$和$\Omega(\bullet)$分别表示loss/risk/objective和prior/regularization/structural knowledge.

$min_{\vec{w}}$的过程即是前面说的optimization，通常是迭代优化的过程。有些可以转化为最优化问题，通过数值优化的方法，如batch的GD/LBFGS，或者onlineSGD／FTRL等进行参数估计；有些无法构造成可解的最优化问题时，转化为概率分布的孤寂问题，通过probabilistic inference来解决，比如用Gibbs Sampling来训练Latent Dirichlet Allocation模型。

无论是数据优化，还是Sampling，都是不断迭代的过程：

$$
\vec{w}^0 = initial \; model \\\
for i \leftarrow 1 to T \\\
\nabla_i = g(\mathcal{D}, \vec{w}^{i-1}) \\\
\vec{w}^i = \vec{w}^{i-1} + \mu(\nabla^i)
$$

每一步迭代做两件事：1. 当前的模型$\vec{w}^{i-1}$在数据集$\mathcal{D}$上的evaluation，得到一个“与最好模型相比可能存在的偏差”量；2. 根据这个偏差量，去修正模型的过程。


大规模机器学习就是解决$\mathcal{D}$和$\vec{w}$的规模非常大的时候所引入的理论上和工程上的问题。“知识体系”可以从两个方面来整理，一方面是宏观上的架构，另一方面时无数微观上的trick。迭代中的那两步决定了宏观上的架构应该是什么样子。两个步骤内的计算和之间的数据交互引入了解决多个问题的技术和trick。

首先：$\mathcal{D}$和$\vec{w}$需要分布在多个计算节点上。

假设数据分布在n个数据节点上，模型分布在m个参数节点上。数据／样本集只与第一步有关，这部分计算应该在数据所在的节点上进行；第二步是对模型的更新，这一步应该在模型所在的参数节点上进行。也就是说，这个架构中有两个角色，每个角色都有自己的计算逻辑。并且在分布式系统中，需要有一套replica的机制来容错。

几乎所有的大规模机器学习平台、系统都可以诶看做由这两个角色构成。在Spark MLLib里，driver program是model node，executor所在的worker node是data node；在Vowpal Wabbit里面，mapper id为0的节点扮演了model node的角色，所有的mapper扮演了data node的角色；在Parameter Server里面，server是model node，worker是data node。其中MLLib和VW的model node都是单节点，而PS可以把模型扩展到多个节点。

其次，节点之间需要传输数据。data node需要在计算前获取当前的模型$\vec{w}^{i-1}$，model node需要得到更新量$\nabla^{i}$.

MLLib通过RDD的treeAggregate接口，VW通过allReduce，这两者其实是相同的方式。节点被分配在一个树状的结构中，每个节点在计算好自己的这部分更新量$\nabla_{k}^{i}$后，汇总它的字节点结果，传给它的父节点。最终，根节点（model node）获取到总的更新量$\nabla^{i} = \sum_k \nabla_k^i$. DMLC中的rabit对这类aggregation数据传输提供了更简洁的接口和更鲁棒的实现。我们用rabit实现了一套multi-task learning去训练不同广告位上广告点击率模型的方案。

而PS则是通过worker（data node）去pull其需要的那部分$\vec{w}^{i-1}$，计算更新量并把自己的那部分$\nabla_k^i$去push给server（model node），server去做对应的模型更新。这个似乎比前面的aggregation要简单，其实不然，当每个worker进度差异特别大时（网络通信／机器资源导致），又引入了新的问题。

所以，第三个问题涉及到并行和一致性。分布式系统中每个节点的状态都不相同，计算的进度也会不一样。当某个data node计算的更新量$\nabla_k^i$对应的是若干轮迭代之前的$\vec{w}^{i-n}$时，它对优化不一定有贡献，甚至可能影响到收敛。

MLLib和VW上的迭代计算，可以看作是同步的。model node获取了所有的data node的$\nabla_k^i$

