## dmlc:ps-lite总结

+ author: zhouyongsdzh@foxmail.com

### 目录

+ ps-lite使用
    + 安装与编译
    + demo
+ ps-lite基本概念
+ ps工作原理
+ [4. PS实战：分布式机器学习](#4.PS实战：分布式机器学习)
    + [4.1. 总控逻辑](#4.1.总控逻辑) 
    + [4.2. Server端逻辑](#4.2.Server端逻辑)：包括优化器，局部模型数据结构，同步／异步逻辑，模型文件；
    + [4.3. Worker端逻辑](#4.3.Worker端逻辑)：包括学习器，训练／测试数据与初始化，与server和scheduler交互信号；
    + [4.4. 通信内容接口](#4.4.通信内容接口)：server，worker实际通信单元接口
    + [4.5. 学习器](#4.5.学习器)：包括：loss, predicter, gradient, updater接口以及模型参数；
    + [4.6. 优化器](#4.6.优化器)：优化算法
    + [4.7. 损失函数](#4.7.损失函数)：实现Loss，用于初始化学习器；
    + [4.8. 评估模块](#)

### ps-lite使用

#### 安装与编译

编译过程：

```bash
# 下载
git clone https://github.com/dmlc/ps-lite
# 进入目录，make过程会自动下载依赖文件（需要联网）
cd ps-lite && make -j4
```
`libps.a`：静态依赖库，在`${dir}/build/`目录下面.
`include`：在`${dir}/include`下面

#### ps demo

```c++
#include "ps/ps.h"using namespace ps;#include <iostream>int num = 0;void ReqHandle(const SimpleData& req, SimpleApp* app) {  CHECK_EQ(req.head, 1);  CHECK_EQ(req.body, "test");  app->Response(req);  ++ num;}int main(int argc, char *argv[]) {  int n = 100;
  // ps::start之前  SimpleApp app(0);         // 定义一个应用  app.set_request_handle(ReqHandle);    // 设置请求处理逻辑  ps::Start();  if (IsScheduler()) {    std::vector<int> ts;    for (int i = 0; i < n; ++i) {      int recver = kScheduler + kServerGroup + kWorkerGroup;      ts.push_back(app.Request(1, "test", recver));
      std::cout << "kScheduler: " << kScheduler 
        << ", kServerGroup: " << kServerGroup 
        << ", kWorkerGroup: " << kWorkerGroup << std::endl;    }    for (int t : ts) {      app.Wait(t);    }  }  ps::Finalize();  CHECK_EQ(num, n);  return 0;}
```

参数含义：

```
kScheduler: 
kServerGroup: 
kWorkerGroup: 
```


<h2 id="4.PS实战：分布式机器学习">4. PS实战：分布式机器学习</h2>

<h3 id="4.1.总控逻辑">4.1. 总控逻辑</h3>

ps的任务通常要初始化以下信息：

```c++
Param param;
ps::Start();                // step1: ps任务开始
LaunchScheduler(param);     // step3: 启动scheduler任务            
LaunchServer(param);        // step2: 启动server任务
LaunchWorker(param);        // step4: 启动worker，任务运行
ps::Finalize();             // step5: ps任务结束
```

<h3 id="4.2.Server端逻辑">4.2. Server端逻辑</h3>

Server端功能：

1. 参数学习与更新逻辑；
2. 参数更新方式控制（异步／同步／多任务ADMM）；
3. 

Server端逻辑如下：

```c++
void LaunchServer() {  if (!ps::IsServer()) return ;  auto server = new ps::KVServer<float>(0);  server->set_request_handle(ps::KVServerDefaultHandle<float>());  RegisterExitCallback([server]() { delete server; })}
```

这里的`ps::KVServer`和`ps::KVServerDefaultHandle`都用了ps默认的参数. 在实际中需要自己写一个`KVServer`类，然后向实例化一个server处理逻辑的类，并填充到`set_request_handle`方法中。

### Worker：worker端逻辑

worker端处理逻辑，针对每一个batch，有以下几步：

1. pull操作（从server端拉取最新模型数据：weight_）
2. 预测打分；（learner完成）
3. 计算梯度；（learner完成，由损失函数决定梯度）
4. push操作（signal: gradient）

因此，worker端需要实现一个**预测器(predictor)、loss函数类(loss)以及梯度计算**.

`(keys, rets, lens) --> map_weight_`这种参数的转化与worker无关，与具体的计算逻辑有关（Learner/Model等）

损失函数决定了梯度更新逻辑，因此在梯度细粒度的推导中，可以用不到模型。或者梯度根据loss来生成（lambda方式）

$$
\quad \min_\vec{w} \; \sum_{(x, y) \in \mathcal{D}} \mathcal{L}(x,y; \vec{w}) \; + \; \lambda_1 \Omega_{1}(\vec{w}) + \; \lambda_2 \Omega_{2}(\vec{w}) \\\
w_j \longleftarrow w_j - \color{purple}{\underbrace{\eta}_{学习率}}  \cdot  \left( \underbrace { \color{orange}{\frac{ \partial }{ \partial w_j} \mathcal{L}(x,y; \vec{w}) }}_{损失函数梯度} + \underbrace {\color{purple}{\lambda_1 \cdot \frac{\partial}{\partial w_j} \Omega_{1}(\vec{w})}}_{\mathcal{l}1正则项梯度} + \underbrace { \color{purple}{\lambda_2 \cdot \frac{\partial}{\partial w_j} \Omega_{2}(\vec{w})}}_{\mathcal{l}2正则项梯度} \right)
$$

$$
w_j \longleftarrow w_j - \color{purple}{\underbrace{\eta}_{学习率}}  \cdot \overbrace { \left( \underbrace { (\text{label - pred}) * \underbrace {\frac{ \partial }{ \partial w_j}f(x)}_{模型表达式梯度} }_{损失函数梯度} + \underbrace {\lambda_1 \cdot \frac{\partial}{\partial w_j} {\Vert w \Vert}_1}_{\mathcal{l}1正则项梯度} + \underbrace { \lambda_2 \cdot \frac{\partial}{\partial w_j} {\Vert w \Vert}_2^2}_{\mathcal{l}2正则项梯度} \right)}^{目标函数梯度}
$$

worker端需要保存的数据有：部分模型权重和梯度值；

> 问题：
> 
> 1. 不同的优化算法，梯度计算方式也不同。像LBFGS这类二阶梯度法是否需要在worker端计算梯度？
> 
> 2. 常见的model是否都对应一个固定的损失函数？比如lr对应log损失，svm对应hinge损失...
> 
> 应该不是一一对应的，比如FM既可以解决回归问题（平方损失），也可以解决分类问题（logit损失），也可以解决排序问题。因此抽出来Learner中间层，包含loss和model;

### Learner

作用：学习器，用于计算predict、loss、gradient和update的计算与更新；不涉及到pull/push操作。包含的信息：

1. 模型：函数表达式，提供predict功能（调用model->Predict函数）
2. Loss：功能是什么呢？与梯度密切相关 
3. 梯度是否与优化算法有关系? 梯度法相关的优化器与worker梯度计算无关；
4. 效果评估，真对每一次epoch；
    5. 它的输入为float(evalset, weight) 

### Model

具体模型的工厂模式，没有具体功能

### Unit

参数计算单元，与model无关。对应线性模型来说，只与field和k有关。具体到下面三个模型：

| 模型参数单元 | field | k | w | length(vector) |
| --- | :--: | --- | --- | :--: |
| LR | 0 | 0 | 1 | `0 * k_ + 1` |
| FM | 1 | k_ | 1 | `1 * k_ + 1` |
| FFM | field_num | k_ | 1 | `field_num * k_ + 1` |

因此在初始化ParamUnit时，只需要field和k即可。因此，完全可以用一个vector来表示。
 
### Updater 

参数更新器，主要功能：

1. 封装优化器（Optimizer）；
2. 数据结构转化：PS本身的结构对优化器透明，转化成开发需要的模型结构；
3. 初始化模型参数：即server中没有，但是worker端传过来的grad对应的key；
    + 需要考虑`超参数`初始化问题，现在默认初始化为0；


虽然参数学习的是模型的参数，但是模型参数单元已经用Unit类表示了，所以Updater处理的对象是Unit，而与模型基本无关了。LR／FM／FFM都是如此。

### Optimizer

优化算法模版类，主要用于初始化具体的优化算法。

### 优化算法（参数学习）

+ 梯度下降算法：
+ FTRL：
+ ALS：
+ MCMC等

优化模型参数，需要知道Unit中参数的长度，即`field_num * k + 1`，而不需要额外知道`wif`信息。


可以参考：
 
## Debug

#### 1. `src/van.cc:58: Check failed: !interface.empty() failed to get the interface`

错误日志：`./include/dmlc/logging.h:235: [08:19:58] src/van.cc:58: Check failed: !interface.empty() failed to get the interface`

原因：重启之后变好了，应该是系统级port和interface被占用？（猜测）

#### 2. `ps/kv_app.h:557: Check failed: (range.size()) == (s.keys.size()) unmatched keys size from one server`

range与s.keys.size大小不一致。提示该错误的根本原因是**没有对keys排序**.

因此，worker与server通信时，需要做以下操作：

```c++
std::vector<ps::Key> keys;
sort(keys.begin(), keys.end());     // 必须要排序，否则报错
```

#### 3. worker向scheduler发送`WORKER_COMPLETE`时，报错：

```c++
[14:41:08] /home/zhouyongsdzh/workspace/openmit/openmit/third_party/dmlc-core/include/dmlc/logging.h:303: [14:41:08] src/van.cc:328: Check failed: obj timeout (5 sec) to wait App 0 readyStack trace returned 5 entries:[bt] (0) /home/zhouyongsdzh/workspace/openmit/openmit/bin/openmit(_ZN4dmlc15LogMessageFatalD2Ev+0x3f) [0x4e88d3][bt] (1) /home/zhouyongsdzh/workspace/openmit/openmit/bin/openmit(_ZN2ps3Van9ReceivingEv+0x1816) [0x59cc86][bt] (2) /usr/lib/x86_64-linux-gnu/libstdc++.so.6(+0xb8c80) [0x7fdaca72cc80][bt] (3) /lib/x86_64-linux-gnu/libpthread.so.0(+0x76ba) [0x7fdacbd586ba][bt] (4) /lib/x86_64-linux-gnu/libc.so.6(clone+0x6d) [0x7fdac9e9282d]terminate called after throwing an instance of 'dmlc::Error'  what():  [14:41:08] src/van.cc:328: Check failed: obj timeout (5 sec) to wait App 0 ready
```

其中，终端提示的错误：

```c++
zhouyongsdzh@ubuntu:~/workspace/openmit/openmit$ ps -ef | grep openmitzhouyon+ 24467  2362  0 15:28 pts/2    00:00:00 bash -c  nrep=0 rc=254 while [ $rc -ne 0 ]; do     export DMLC_NUM_ATTEMPT=$nrep     /home/zhouyongsdzh/workspace/openmit/openmit/bin/openmit /home/zhouyongsdzh/workspace/openmit/openmit/conf/openmit.conf     rc=$?;     nrep=$((nrep+1)); done zhouyon+ 24471  2362  0 15:28 pts/2    00:00:00 bash -c  nrep=0 rc=254 while [ $rc -ne 0 ]; do     export DMLC_NUM_ATTEMPT=$nrep     /home/zhouyongsdzh/workspace/openmit/openmit/bin/openmit /home/zhouyongsdzh/workspace/openmit/openmit/conf/openmit.conf     rc=$?;     nrep=$((nrep+1)); done zhouyon+ 24472  2362  0 15:28 pts/2    00:00:00 bash -c  nrep=0 rc=254 while [ $rc -ne 0 ]; do     export DMLC_NUM_ATTEMPT=$nrep     /home/zhouyongsdzh/workspace/openmit/openmit/bin/openmit /home/zhouyongsdzh/workspace/openmit/openmit/conf/openmit.conf     rc=$?;     nrep=$((nrep+1)); done zhouyon+ 24476  2362  0 15:28 pts/2    00:00:00 bash -c  nrep=0 rc=254 while [ $rc -ne 0 ]; do     export DMLC_NUM_ATTEMPT=$nrep     /home/zhouyongsdzh/workspace/openmit/openmit/bin/openmit /home/zhouyongsdzh/workspace/openmit/openmit/conf/openmit.conf     rc=$?;     nrep=$((nrep+1)); done zhouyon+ 24480  2362  0 15:28 pts/2    00:00:00 bash -c  nrep=0 rc=254 while [ $rc -ne 0 ]; do     export DMLC_NUM_ATTEMPT=$nrep     /home/zhouyongsdzh/workspace/openmit/openmit/bin/openmit /home/zhouyongsdzh/workspace/openmit/openmit/conf/openmit.conf     rc=$?;     nrep=$((nrep+1)); done zhouyon+ 24564 24476  1 15:28 pts/2    00:00:01 /home/zhouyongsdzh/workspace/openmit/openmit/bin/openmit /home/zhouyongsdzh/workspace/openmit/openmit/conf/openmit.confzhouyon+ 24565 24467  1 15:28 pts/2    00:00:01 /home/zhouyongsdzh/workspace/openmit/openmit/bin/openmit /home/zhouyongsdzh/workspace/openmit/openmit/conf/openmit.confzhouyon+ 24572 24472  1 15:28 pts/2    00:00:01 /home/zhouyongsdzh/workspace/openmit/openmit/bin/openmit /home/zhouyongsdzh/workspace/openmit/openmit/conf/openmit.confzhouyon+ 24576 24471  1 15:28 pts/2    00:00:01 /home/zhouyongsdzh/workspace/openmit/openmit/bin/openmit /home/zhouyongsdzh/workspace/openmit/openmit/conf/openmit.confzhouyon+ 24580 24480  1 15:28 pts/2    00:00:01 /home/zhouyongsdzh/workspace/openmit/openmit/bin/openmit /home/zhouyongsdzh/workspace/openmit/openmit/conf/openmit.conf
```

此外，worker.cc里面部分代码只有这样，才能运行结束：

```c++
  //kv_worker_->Request(signal::WORKER_COMPLETE, "worker", ps::kScheduler);    // message to tell server job finish    std::vector<ps::Key> keys(1,1);  std::vector<float> vals(1,1);  std::vector<int> lens(1,1);  kv_worker_->Wait(      kv_worker_->Push(keys, vals, lens, mit::signal::FINISH));
```
