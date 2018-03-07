## 主要包括PS和MPI两类计算框架，重点突出
+ PS的计算过程（异步），基于此实现了哪些模型与算法；
    + ps-lite工作原理、工作过程各模块功能；
    + ps同步方式：ASP/BSP/SSP
    + ps大规模集群遇到的问题
        + 通信超时，主要原因是高频通信&&高并发请求，导致丢包；
        + Worker Final退出时异常，表现在resender仍然在工作；
    + 《大规模机器学习技术与OpenMIT：参数服务器详解》 

+ MPI分布式计算工作原理
    + 通信ranker分配，allreduce／broadcast
    + rabit作为轻量级MPI实现，容错机制；
    + 在OpenMIT中，以MPI作为计算框架的机器学习模型，是工作在ADMM算法框架基础上的；
    + ADMM算法框架 详细见《大规模机器学习技术与OpenMIT：ADMM算法框架》


