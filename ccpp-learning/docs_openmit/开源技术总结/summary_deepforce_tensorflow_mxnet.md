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
        + [1.2.3. Graph](#1.2.3.Graph)

---

### MxNet

+ [1. 先睹为快](#1.先睹为快)
    + [1.1. Installing MxNet on Linux](#1.1.Installing MxNet on Linux) 
    + [1.2. Installing TensorFlow for C](#1.2.Installing TensorFlow for C)
    + [1.3. C++使用MxNet](#1.3.C++使用MxNet)

# TensorFlow 

<h2 id="1.先睹为快">1. 先睹为快</h2>

<h3 id="1.1.Installing TensorFlow on Ubuntu">1.1. Installing TensorFlow on Ubuntu</h3>

以Ubuntu CPU环境为例，参考官网：https://www.tensorflow.org/install/install_linux

TensorFlow在Linux环境下提供了4种安装方式，分别是：

+ virtualenv
+ native pip
+ Dockor

**安装**：以virtualenv方式为例：

```bash
sudo apt-get install python-pip python-dev python-virtualenv
virtualenv --system-site-packages targetDirectory 
source ${targetDirectory}/bin/activate  # 激活python虚拟环境
pip install --upgrade tensorflow        # 安装tensorflow python版
```

> 1. 以上步骤安装失败，重复尝试几次即可；
> 2. `targetDirectory`目录：`~/software/tensorflow`. 

**使用方式**

+ 每次要使用tensorflow时，都需要激活虚拟环境。即`source ${targetDirectory}/bin/activate`；
+ 使用完毕后，使用`deactivate`命令 退出虚拟环境；

Uninstalling：`rm -rf ${targetDirectory}`

> 本地Ubuntu安装目录：`targetDirectory: ~/software/tensorflow`

**安装验证**：virtualenv模式

```py
(tensorflow) xyz@ubuntu:~/software$ pythonPython 2.7.12 (default, Nov 19 2016, 06:48:10) [GCC 5.4.0 20160609] on linux2Type "help", "copyright", "credits" or "license" for more information.>>> import tensorflow as tf>>> hello = tf.constant('Hello, TensorFlow!')>>> sess = tf.Session()2017-08-04 17:47:57.610547: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.2017-08-04 17:47:57.610595: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.2017-08-04 17:47:57.610604: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.2017-08-04 17:47:57.610610: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.2017-08-04 17:47:57.610615: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.>>> print(sess.run(hello))Hello, TensorFlow!
```
说明安装成功。

<h3 id="1.2.Installing TensorFlow for C">1.2. Installing TensorFlow for C</h3>

安装TensorFlow C语言接口，参考官网：https://www.tensorflow.org/install/install_c

TensorFlow提供4种语言的安装方式，分别是Python, Java, Go和C语言，这里以安装C语言接口为例（使用较多，python为默认接口）。

参考官网，并作了修改，如下：

```bash
TF_TYPE="cpu"
OS="linux"
TARGET_DIRECTORY="/usr/local"
wget "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-${OS}-x86_64-1.2.1.tar.gz
sudo tag -zxvf libtensorflow-${TF_TYPE}-${OS}-x86_64-1.2.1.tar.gz -C $TARGET_DIRECTORY
```
这里`TARGET_DIRECTORY`可以指向其它目录，但是需要加入库路径中:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mydir/lib # For Linux only
```

接口测试，编辑C程序：

```c
#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main() {
  printf(“Hello from TensorFlow C library version %s\n”, TF_Version());
  return 0;
}
```
编译

```bash
gcc -I/usr/local/include -L/usr/local/lib hello_tf.c -ltensorflow   # 编译
./a.out
Hello from TensorFlow C library version 1.2.1   # 输出结果
```

<h3 id="1.3.C++使用TF">1.3. C++使用TF</h3>

参考:[Tensorflow C++ API调用预训练模型和生产环境编译](http://deepnlp.org/blog/tensorflow-cpp-build-for-production/)

安装TensorFlow编译所需环境：编译工具bazel. Ubuntu安装过程：

```bash
# 1. 安装java8版本
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update 
sudo apt-get install oracle-java8-installer     // 安装JDK8版本
# 2. Add Bazel distribution URI as a package source
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
# 3. 安装和更新bazel
sudo apt-get update && sudo apt-get install bazel
# sudo apt-get upgrade bazel // 用于更新版本
which bazel
/usr/bin/bazel  // 安装成功
```

> Ubuntu上安装bazel 参考官网：https://docs.bazel.build/versions/master/install-ubuntu.html . 官网也介绍了bazel的二进制安装方式（Install using binary installer） 
> 
参考：https://www.tensorflow.org/get_started/get_started


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
In file included from /opt/meituan/rec-ads/mltools/mltools-deepapp/mxnet/mshadow/mshadow/tensor.h:16:0,
                 from include/mxnet/base.h:31,
                 from src/operator/nn/./../mxnet_op.h:29,
                 from src/operator/nn/./softmax-inl.h:29,
                 from src/operator/nn/softmax.cc:24:
/opt/meituan/rec-ads/mltools/mltools-deepapp/mxnet/mshadow/mshadow/./base.h:143:23: fatal error: cblas.h: No such file or directory
     #include <cblas.h>
                       ^
提示openblas下面的cblas.h和-lopenblas不存在，［临时方案］需要修改mshadow/make/mshadow.mk文件
```

整体安装脚本：

```bash
#!/bin/bash

set -x

GCC_HOME=/opt/meituan/gcc-4.8.2

export PATH=${GCC_HOME}/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$LD_LIBRARY_PATH
echo $PATH
echo ${LD_LIBRARY_PATH}

# OpenBLAS
OPENBLAS_DIR=/opt/meituan/rec-ads/mltools/mltools-deepapp/software/OpenBLAS_lib
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






