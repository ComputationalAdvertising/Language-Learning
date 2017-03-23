## dmlc:ps-lite总结

+ author: zhouyongsdzh@foxmail.com

### 目录

+ ps-lite使用
    + 安装与编译
    + demo
+ ps-lite基本概念
+ ps工作原理
+ [4. PS实战：分布式机器学习](#4.PS实战：分布式机器学习)


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

+ 总控逻辑

ps的任务通常要初始化以下信息：

```c++
Param param;
LaunchServer(param);        // step1: 启动server任务
LaunchScheduler(param);     // step2: 启动scheduler任务
ps::Start();                // step3: ps任务开始
// TODO                 
RunWorker(param);           // step4: 任务运行
ps::Finalize();             // step5: ps任务结束
```

其中，`//TODO`部分会

+ 如何LaunchServer基于param?

Server端的通常逻辑是如下：

```c++
void LaunchServer() {  if (!ps::IsServer()) return ;  auto server = new ps::KVServer<float>(0);  server->set_request_handle(ps::KVServerDefaultHandle<float>());  RegisterExitCallback([server]() { delete server; })}
```

这里的`ps::KVServer`和`ps::KVServerDefaultHandle`都用了ps默认的参数. 在实际中需要自己写一个`KVServer`类，然后向实例化一个server处理逻辑的类，并填充到`set_request_handle`方法中。

### Worker端逻辑

worker端处理逻辑主要分为两步：

1. 预测打分；
2. 计算loss损失和梯度；

因此，worker端需要实现一个**预测器(predictor)**和**loss函数类(loss)**

> 问题：
> 
> 1. 不同的优化算法，梯度计算方式也不同。像LBFGS这类二阶梯度法是否需要在worker端计算梯度？
> 
> 2. 常见的model是否都对应一个固定的损失函数？比如lr对应log损失，svm对应hinge损失...
> 
> 应该不是一一对应的，比如FM既可以解决回归问题（平方损失），也可以解决分类问题（logit损失），也可以解决排序问题。

worker端更新数据的逻辑：

```c++
auto batch = Batch.Value();
for (int i = 0; i < batchSize; ++i) {
    ...
}
```

## Debug

1. `src/van.cc:58: Check failed: !interface.empty() failed to get the interface`

错误日志：`./include/dmlc/logging.h:235: [08:19:58] src/van.cc:58: Check failed: !interface.empty() failed to get the interface`

原因：重启之后变好了，应该是系统级port和interface被占用？（猜测）
