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

# TensorFlow 

<h2 id="1.TensorFlow入门">1.TensorFlow入门</h2>

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
>>> print(tf.__version__)   // 查看tensorflow版本
1.2.1
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


<h3 id="1.2.核心概念">1.2.核心概念</h3>

Tensorflow是基于图（`Graph`）的计算系统。而图的节点则是由操作（`Operation`）来构成的，而图的各个节点之间则是由张量（`Tensor`）作为边来连接在一起的。所以Tensorflow的计算过程就是一个Tensor流图。Tensorflow的图则是必须在一个`Session`中来计算。 我们先看下`Session`是什么？

---
<h4 id="1.2.1.Session">1.2.1.Session</h4>
---

TensorFlow中的Session（会话）的物理意义一句话总结为： 

> **持有并管理TensorFlow程序运行时的所有资源**

**调用Session的方式**

方式一：明确的调用Session的生成函数和关闭Session函数

```py
import tensorflow as tf
# create a sessionsess = tf.Session()# use a session to run taskhello = tf.constant("Hello TensorFlow")#print(sess.run(hello))print(hello.eval(session = sess))# close session sess.close()
```
调用这种方式时，要明确调用Session.close()，以释放资源。当程序异常退出时，关闭函数就不能被执行，从而导致资源泄露。其中第6行与第7行等价。

方式二： 上下文管理机制自动释放所有资源

```py
import tensorflow as tf
# 创建session，并通过上下文机制管理器管理该Session
hello = tf.constant("Hello TensorFlow")with tf.Session() as sess:  result = sess.run(hello)  print(result)
# 无需再调用"sess.close()", 退出with statement时，自动关闭会话和释放资源
```

还有一种Session的默认方式：

```py
sess = tf.Session()with sess.as_default():  # hello as tensor  print(hello.eval())
```

在交互式环境下，通过设置默认会话的方式来获取Tensor的取值更为方便，调用函数`tf.InteractiveSession()` 将省去产生的Session注册到默认会话的过程。

建议使用方式二生成会话，这三种方式都可以通过`ConfigProto Protocol Buffer`来配置需要生成的会话，如**并行线程数、GPU分配策略、运算超时时间**等参数，最常用的两个参数是：`allow_soft_placement`和`log_device_placement`.

`ConfigProto`配置方法：

```pyconf = tf.ConfigProto(allow_soft_placement = True,                       log_device_placement = True)sess = tf.Session(config = conf)print(hello.eval(session = sess))
```

在Unbuntu上打出来的信息：

```bash
Device mapping: no known devices.2017-10-17 08:55:28.810144: I tensorflow/core/common_runtime/direct_session.cc:265] Device mapping:Const_1: (Const): /job:localhost/replica:0/task:0/cpu:02017-10-17 08:55:28.810471: I tensorflow/core/common_runtime/simple_placer.cc:847] Const_1: (Const)/job:localhost/replica:0/task:0/cpu:0Const: (Const): /job:localhost/replica:0/task:0/cpu:02017-10-17 08:55:28.810488: I tensorflow/core/common_runtime/simple_placer.cc:847] Const: (Const)/job:localhost/replica:0/task:0/cpu:0Hello TensorFlow
```

`allow_soft_placement`：boolean，一般设置True，很好的支持多GPU或者不支持GPU时自动将运算放在CPU上。

`log_device_placement`：boolean，为True时日志将会记录每个节点被安排在了哪个设备上以方便调试（生产环境通常设置为False 以减少日志量）。

---
<h4 id="1.2.2.Tensor">1.2.2. Tensor</h4>
---

Tensor是所有深度学习框架（TF、MxNet等）最核心的组件。在TensorFlow中，将数据表示为张量（Tensor）。

**`Tensor是一个多维数组。标量：0阶张量；向量：1阶张量；矩阵：2阶张量；更高维：n阶张量`**


> TensorFlow提供非常丰富的API，最底层的API—TensorFlow Core—提供了完整的编程控制接口。更高级别的API则是基于TensorFlow Core API之上，并且非常容易学习和使用，更高层次的API能够使一些重复的工作变得更加简单。比如tf.contrib.learn帮助你管理数据集等。

> 将各种各样的数据抽象成张量表示，然后再输入神经网络模型进行后续处理是一种非常必要且高效的策略。因为如果没有这一步骤，我们就需要根据各种不同类型的数据组织形式定义各种不同类型的数据操作，这会浪费大量的开发者精力。更关键的是，当数据处理完成后，我们还可以方便地将张量再转换回想要的格式。例如Python NumPy包中**numpy.imread和numpy.imsave**两个方法，分别用来将图片转换成张量对象（即代码中的Tensor对象），和将张量再转换成图片保存起来。

**Rank (秩)**


Rank本意是矩阵的秩。不过Tensor Rank与Matrix Rank意义不同，前者的意义看起来更像是维度，比如Rank=1是向量，Rank＝2是矩阵，Rank＝0是标量（一个值）。

**`A tensor's rank is its number of dimensions.`**

>
>3 # a rank 0 tensor; this is a scalar with shape []   // 没有值 表示标量 <br>
[1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3] <br>
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3] <br>
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

Shape (形状)

**`Shape就是Tensor在各个维度上的长度组成的数组。`**

> 
| Rank | 0 | 1 | 2 | 3 | ... |
| :--: | --- | --- | --- | --- | --- |
| Shape | `[]` | `[a]` | `[a,b]` | `[a,b,c]` | ... |


**[TensorFlow中的Tensor操作](http://www.cnblogs.com/wuzhitj/p/6431381.html)**

+ 数据类型转换（casting）

	| 操作 | 描述 |
	| :-- | :--: | 
	| `tf.string_to_number`<br>`(string_tesnor, out_type=None, name=None)` | 
	| `tf.to_double(x, name='ToDouble')` |
	
+ 形状操作（shaping）

	| 操作 | 描述 |
	| :-- | --- |
	| `tf.shape(input, name=None)` | 返回Tensor的shape <br> `t is [[[1,1,1], [2,2,2]], [[3,3,3], 4,4,4]]]`<br> `tf.shape(t) --> [2,2,3]` |
	| `tf.size(input, name=None)` | 返回Tensor的元素数量<br>`t is [[[1,1,1], [2,2,2]], [[3,3,3], 4,4,4]]]`<br> `tf.size(t) --> 12` |
	| `tf.rank(input, name=None)` | 返回Tensor的rank（维度）<br>`t is [[[1,1,1], [2,2,2]], [[3,3,3], 4,4,4]]]`<br> `tf.rank(3) --> 3` |
	| `tf.reshape(tensor, shape, name=None)` | 改变Tensor的形状 .... |
	
+ 切片与合并（slicing & joining）

	| 操作 | 描述 |
	| :-- | --- |
	| `tf.slice(input_, begin, size, name=None)` |

---	
<h4 id="1.2.3.Operation">1.2.3. Operation</h4>
---

一个Operation就是TensorFlow Graph中的一个计算节点。它接受0个或多个Tensor对象作为输入，然后产生0个或多个Tensor对象作为输出。

```py
import tensorflow as tf# ============ create op ===============# creat a constant op. shape: 1*2matrix1 = tf.constant([[3., 3.]])# create a constant op. shape: 2*1matrix2 = tf.constant([[2.], [2.]])# create a matrix multiple op product = tf.matmul(matrix1, matrix2)# ============ create session ==========with tf.Session() as sess:  result = sess.run(product)  print(result)
```

当一个Graph加载到一个Session中，则可以调用`Session.run(product)`来执行op，或者调用`op.run()`来执行（`product.run()`是`tf.get_default_session().run()`的缩写）。

---
<h4 id="1.2.4.Graph">1.2.4. Graph</h4>
---

```py
import tensorflow as tf# ============ graph: use method 1 ===============c = tf.constant(value=1)assert c.graph is tf.get_default_graph()print(c.graph)print(tf.get_default_graph())# ============ graph: use method 2 ===============g = tf.Graph()print("g: ", g)with g.as_default():  d = tf.constant(value=2)  print(d.graph)g2 = tf.Graph()print("g2: ", g2)g2.as_default()e = tf.constant(value=10)print(e.graph)
```
运行结果：

```py
<tensorflow.python.framework.ops.Graph object at 0x7fc7464bd490><tensorflow.python.framework.ops.Graph object at 0x7fc7464bd490>('g: ', <tensorflow.python.framework.ops.Graph object at 0x7fc7464c6750>)<tensorflow.python.framework.ops.Graph object at 0x7fc7464c6750>('g2: ', <tensorflow.python.framework.ops.Graph object at 0x7fc732eaea10>)<tensorflow.python.framework.ops.Graph object at 0x7fc7464bd490>
```

`Graph`的三种用法：

+ **默认图**： 如`use method1`中使用的是默认图(通过`tf.get_default_graph()`)；
+ **上下文管理器**：如`use method2`使用到`Graph.as_default()`的上下文管理器（context manager），它能够在上下文里面覆盖默认的图。
    
    > 创建一个新图g，然后把它设置为默认。接下来的操作就不是在默认图中，而是在g中了（可以理解为g就是新的默认图了）。<br>
    > 注意：最后的变量e不是定义在with语句里面的，是在最开始的那个图中（不在上下文管理器的图中）。


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






