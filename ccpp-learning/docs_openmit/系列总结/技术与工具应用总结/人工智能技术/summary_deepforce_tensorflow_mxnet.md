## DeepForce 

+ author: zhouyongsdzh
+ date: 2016 Q2

### TensorFlow

+ [1. TensorFlow入门](#1.TensorFlow入门)
    + [1.1. 安装与使用](#1.1.安装与示例) 
        + [1.1.1. Installing TensorFlow on Ubuntu](#1.1.Installing TensorFlow on Ubuntu) 
        + [1.1.2. Installing TensorFlow for C](#1.2.Installing TensorFlow for C)
        + [1.1.3. C++使用TF](#1.3.C++使用TF)
    + [1.2. 核心概念](#1.2.核心概念)
        + [1.2.1. Session](#1.2.1.Session) 
        + [1.2.2. Tensor](#1.2.2.Tensor)
        + [1.2.3. Operation](#1.2.3.Operation)
        + [1.2.4. Graph](#1.2.4.Graph)

---

### MxNet

+ [1. 先睹为快](#1.先睹为快)
    + [1.1. Installing MxNet on Linux](#1.1.Installing MxNet on Linux) 
    + [1.2. Installing TensorFlow for C](#1.2.Installing TensorFlow for C)
    + [1.3. C++使用MxNet](#1.3.C++使用MxNet)



# MxNet

<h2 id="1.先睹为快mxnet">1. 先睹为快</h2>

>
1. [MxNet GitHub](https://github.com/apache/incubator-mxnet.git): 源码地址
2. [MxNet Install](http://mxnet.io/get_started/install.html): 介绍在各种环境下的安装方法。
3. [MxNet Architecture](http://mxnet.io/architecture/index.html): 体系结构，包括系统概况、DL编程风格（Symbolic和Imperative）、DL依赖引擎、内存优化、**数据加载模块**等；
4. [MxNet Tutorials](http://mxnet.io/tutorials/index.html): 介绍了DL基础知识和MxNet实现深度学习任务。给出了NDArray/Symbol/Module/Iterators核心概念解释，以及LR、图像识别领域demo；
5. [MxNet Example](https://github.com/apache/incubator-mxnet/tree/master/example): 给出了MxNet用于深度学习的各种示例，包MLP、RNN、CNN、DAG、LSTM等；
6. [MxNet API](http://mxnet.io/api/c++/index.html)：支持Python/Scala/R/C++/Java等接口，namespace/classes/文件结构 介绍的非常清楚；
7. [MxNet Efficient Data Loaders](http://mxnet.io/architecture/note_data_loading.html): MxNet高效的数据加载器，需要认真读。

<h3 id="1.1.Installing MxNet on Linux">1.1. Installing MxNet on Linux</h3>

mxnet安装官网：http://mxnet.io/get_started/install.html#validate-mxnet-installation

> 经实践，是可行的。

这里本人安装环境(Ubuntu)选择：`Linux + Python + CPU + Build from Source` 方式

查看GPU使用情况：`nvidia-smi`

本人安装环境是在sudo权限，在开发机(Centos)上没有sudo权限，因此**MxNet的上游依赖需要手动安装**。

+ 安装OpenBLAS和LAPACK库

```bash
# 安装OpenBLAS. 参考官网：https://github.com/xianyi/OpenBLAS/wiki/Installation-Guide
# step1: clone
git clone https://github.com/xianyi/OpenBLAS.git   # clone源码
# step2: 编译
1). make                # 默认编译的是支持多线程、无fortran库支持
2). make FC=gfortran    # 多线程支持gfortran 
# 如果执行make FC=gfortran报错下面，说明机器未安装gfortran. Ubuntu下可使用sudo apt-get install gfortran安装
OpenBLAS  make[2]: gfortran: Command not found  
# step3: 安装 （PREFIX指定路径）
make install PREFIX=~/mltools/mltools-deepapp/software/OpenBLAS_lib
```

```bash
# 安装LAPACK
# step1: clone
git clone https://github.com/Reference-LAPACK/lapack.git
# step2: 编译 (必须在gfortran环境下安装，如果没有需要源码安装高版本gcc)
export PATH=~/gcc-4.8.2/bin:$PATH
export LD_LIBRARY_PATH=~/gcc-4.8.2/lib64:$LD_LIBRARY_PATH
# 测试gfortran是否存在 gfortran -v
cd lapack && make           # 该过程比较耗时
# 编译完成后 lapack目录会出现：liblapack.a, librefblas.a, libtmglib.a 三个静态文件
```
+ 安装OpenCV

+ 安装MxNet

```bash
make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas
# 报错1
In file included from /opt/xxx/rec-ads/mltools/mltools-deepapp/mxnet/mshadow/mshadow/tensor.h:16:0,
                 from include/mxnet/base.h:31,
                 from src/operator/nn/./../mxnet_op.h:29,
                 from src/operator/nn/./softmax-inl.h:29,
                 from src/operator/nn/softmax.cc:24:
/opt/xxx/rec-ads/mltools/mltools-deepapp/mxnet/mshadow/mshadow/./base.h:143:23: fatal error: cblas.h: No such file or directory
     #include <cblas.h>
                       ^
提示openblas下面的cblas.h和-lopenblas不存在，［临时方案］需要修改mshadow/make/mshadow.mk文件
```

整体安装脚本：

```bash
#!/bin/bash

set -x

GCC_HOME=/opt/xxx/gcc-4.8.2

export PATH=${GCC_HOME}/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$LD_LIBRARY_PATH
echo $PATH
echo ${LD_LIBRARY_PATH}

# OpenBLAS
OPENBLAS_DIR=/opt/xxx/rec-ads/mltools/mltools-deepapp/software/OpenBLAS_lib
export LD_LIBRARY_PATH=$OPENBLAS_DIR/lib:$LD_LIBRARY_PATH
export PATH=$OPENBLAS_DIR/bin:$PATH
export PLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$OPENBLAS_DIR/include

# lapack
#cd software/lapack && make

# opencv

# mxnet
export LD_LIBRARY_PATH
cd mxnet.exp
make -j $(nproc) USE_OPENCV=0 USE_BLAS=openblas
```

<h3 id="1.3.C++使用MxNet">1.3. C++使用MxNet</h3>






