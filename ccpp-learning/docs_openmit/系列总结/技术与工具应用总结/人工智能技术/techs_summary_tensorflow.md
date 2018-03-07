## TensorFlow应用与总结

+ author: zhouyongsdzh@foxmail.com

目录 
--
### 第一部分：TensorFlow框架
+ [1. TensorFlow入门](#1.TensorFlow入门)
    + [1.1. 安装与使用](#1.1.安装与示例) 
        
        | Install | [Installing TensorFlow on Ubuntu](#1.1.Installing TensorFlow on Ubuntu) | [Installing TensorFlow for C](#1.2.Installing TensorFlow for C) |
        | :--: | :--: | :--: |
        | Use | [C++使用TF](#1.3.C++使用TF) |
    + [1.2. 核心概念](#1.2.核心概念)

        | [Session](#1.2.1.Session)  | [Tensor](#1.2.2.Tensor) | [Operation](#1.2.3.Operation) | [Graph](#1.2.4.Graph) | [Variable](#1.2.5.Variable) |
        | --- | --- | --- | --- | --- |
    + [1.3. IO](#1.3.IO)
+ [2. Embedding](#2.TensorFlow Embedding)
    + [2.1. `embedding_lookup_sparse`](#2.1.embedding_lookup_sparse)

### 第二部分: TensorFlow Serving

+ 1. Serving入门
    + 1.1. 安装与使用 

--
## 第一部分：TensorFlow框架

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

+ TF源码编译

安装TensorFlow编译所需环境：**bazel**。 Ubuntu安装过程：

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

+ 编译TF 

```bash
# 1. 下载
git clone --recursive https://github.com/tensorflow/tensorflow
cd tensorflow
# 2. 编译生成.so文件, 编译C++ API的库 
bazel build //tensorflow:libtensorflow_cc.so
# 编译C API的库 ［可选］
bazel build //tensorflow:libtensorflow.so
# 编译成功后，在tensorflow根目录bazel-bin/tensorflow/目录下会出现libtensorflow.so/libtensorflow_cc.so/libtensorflow_framework.so文件
# 3. 生成必要的依赖头文件, 执行
sh tensorflow/contrib/makefile/build_all_linux.sh
# 可以看到tensorflow/contrib/makefile/gen/protobuf/include目录，需要包含在cmake的include_directories()里.
# tensorflow和bazel-genfiles也需要放在include_dicectories目录下。
```

> 如果编译失败，提示错误码为`4`，应该是swap空间问题，重复编译几次即可。

**CMake调用TF动态库和头文件**

遇到的编译错误：

> 1. `tensorflow/tensorflow/core/platform/default/mutex.h:25:22: fatal error: nsync_cv.h: No such file or directory`
> 需要在CMakeLists.txt中的配置`include_directories`添加：`/home/zhouyongsdzh/software/tensorflow-family/tensorflow/bazel-tensorflow/external/nsync/public`
> 
> 2. `/tensorflow/third_party/eigen3/unsupported/Eigen/CXX11/Tensor:1:42: fatal error: unsupported/Eigen/CXX11/Tensor: No such file or directory`
> 错误原因： 没有安装或找到Eigen的路径。
> 需要在CMakeLists.txt中的配置`include_directories`添加：`${tf_root_dir}/tensorflow/tensorflow/contrib/makefile/downloads/eigen`
> 
> 3. `libtensorflow_cc.so: undefined reference to tensorflow::_OptimizerOptions_default_instance_`
>  错误原因：没有`tensorflow_framework`动态库 （与`libtensorflow_cc.so`在同一个目录）
> 解决方案：将`libtensorflow_framework.so`添加到`include_libraries`里，同时配置`list(APPEND tfserving_LINKER_LIBS tensorflow_framework)`
> 
> 4. `unsupported/Eigen/CXX11/Tensor:1:42: error: #include nested too deeply`
> 错误原因：



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

> **将各种各样的数据抽象成张量表示，然后再输入神经网络模型进行后续处理是一种非常必要且高效的策略**。因为如果没有这一步骤，我们就需要根据各种不同类型的数据组织形式定义各种不同类型的数据操作，这会浪费大量的开发者精力。更关键的是，当数据处理完成后，我们还可以方便地将张量再转换回想要的格式。例如Python NumPy包中**numpy.imread和numpy.imsave**两个方法，分别用来将图片转换成张量对象（即代码中的Tensor对象），和将张量再转换成图片保存起来。

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
	
注意：以上操作必须在`sess.run()`下面执行才能看到信息。这也是tf任务执行的基本思想。 

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
    
---
<h4 id="1.2.5.Variable">1.2.5. Variable</h4>
---

TensorFlow中的图变量，跟我们平时所接触的一般变量在用法上有很大的差异。

图变量的两种定义方法：`tf.Variable()`和`tf.get_variable()`

**`tf.Variable`的用法**

[`tf.Variable(initial_value, trainable=True, collections=None, validate_shape=True, name=None)`]()

使用方式：

```py
import tensorflow as tf import numpy as np# ============= tf.Variable =============tensor1 = tf.constant(np.arange(1, 13), shape=[2,3,2], dtype=tf.float32)a1 = tf.Variable(tensor1, dtype=tf.float32, name='a1')a2 = tf.Variable(initial_value=tf.random_normal(shape=[2,3], mean=0, stddev=1), name='a2')a3 = tf.Variable(tf.ones(shape=[2, 3]), name='a3')a4 = a2.assign(tf.constant(np.arange(1,7), shape=[2,3], dtype=tf.float32))# sessionwith tf.Session() as sess:  sess.run(tf.global_variables_initializer())  print('var(a1) info:\n{}, type:{}, shape: {}'.format(sess.run(a1), type(a1), sess.run(tf.shape(a1))))  print('var(a2) info:\n{}'.format(a2.eval(sess)))  print('var(a3) info:\n{}'.format(sess.run(a3)))  print('var(a4) info:\n{}'.format(sess.run(a4)))
```

**`tf.name_scope()`和`tf.variable_scope()`的用法**

先看一个`tf.name_scope()`的例子：

```py
print("\n[INFO] tf.name_scope ....")with tf.name_scope("name_scope_hello") as name_scope:  ns1 = tf.get_variable('ns1', shape=[2, 10], dtype=tf.float32)  print name_scope   print type(name_scope)  print ns1.name  var2 = tf.Variable(5, name='var2')  print var2.name  print("scope_name: %s" % tf.get_variable_scope().original_name_scope) ```

输出结果：

```py
[INFO] tf.name_scope ....name_scope_hello/<type 'str'>ns1:0name_scope_hello/var2:0scope_name: 
```

可以看到： 

1. `tf.name_scope()`返回的是string类型，'name_scope_hello/'
2. `tf.get_variable()`方法定义的变量名没有`name_scope_hello/`前缀，而`tf.Variable`方法则有；
3. `tf.get_variable_scope()`的`original_name_scope`为空。

看一个`tf.variable_scope()`例子：

```pywith tf.variable_scope('variable_scope_hello') as variable_scope:  vs1 = tf.get_variable('vs1', shape=[2,10], dtype=tf.float32)  print variable_scope   print type(variable_scope)  print variable_scope.name   print tf.get_variable_scope().original_name_scope   print vs1.name
  vs2 = tf.Variable([1,2], dtype=tf.int32, name='vs2')  print vs2.name  with tf.variable_scope('xxxx') as x_scope:    print tf.get_variable_scope().original_name_scope 
```

输出结果：

```py
<tensorflow.python.ops.variable_scope.VariableScope object at 0x7fd492e9c0d0><class 'tensorflow.python.ops.variable_scope.VariableScope'>variable_scope_hellovariable_scope_hello/variable_scope_hello/vs1:0
variable_scope_hello/vs2:0variable_scope_hello/xxxx/
```

可以看到：

1. `tf.variable_scope()`返回的类型是一个op对象； 
2. `variable_scope`下定义的variable的name都加上了`variable_scope_hello／`前缀(`tf.get_variable`和`tf.Variable`). 注意与`name_scope`的区别；

看一个二者结合的例子： 

```pyprint('\n[INFO] tf.name_scope && tf.variable_scope ....')with tf.name_scope('name1') as name:  with tf.variable_scope('var1') as var:    nv1 = tf.get_variable('nv1', shape=[2])    add = tf.add(nv1, [3])print nv1.nameprint add.name with tf.Session() as sess:  sess.run(tf.global_variables_initializer())  print sess.run(nv1)  print sess.run(add) 
```

输出结果：

```py
var1/nv1:0name1/var1/Add:0[ 1.17539227  1.19980586][ 4.17539215  4.19980574]
```

可以看出：`variable scope`和`name scope`都会给op（如`tf.Variable`/`tf.add`）的name加上前缀。而非op(如`tf.get_variable`)的name只会加上`variable scope`的前缀。

**name_scope可以用来干什么?**

典型的 TensorFlow 可以有数以千计的节点，如此多而难以一下全部看到，甚至无法使用标准图表工具来展示。为简单起见，我们为op/tensor名划定范围，并且可视化把该信息用于在图表中的节点上定义一个层级。默认情况下， 只有顶层节点会显示。下面这个例子使用tf.name_scope在hidden命名域下定义了三个操作:

```py
# ============= what can tf.name_scope to do? =============print('\nwhat can tf.name_scope to do? ....')with tf.name_scope('hiddle') as scope:  a = tf.constant(10, name='alpha')  W = tf.Variable(tf.random_normal([1,2], -1.0, 1.0), dtype=tf.float32, name='weights')  b = tf.Variable(tf.zeros([1]), name='bias')  print a.name  print W.name  print b.name
```
输出结果：

```
what can tf.name_scope to do? ....hiddle/alpha:0hiddle/weights:0hiddle/bias:0
```

+ 小结：**`name_scope`是给`op_name`加前缀, `variable_scope`是给`get_variable()`创建的变量的名字加前缀**.

**`tf.name_scope(None)`有清除`name space`的作用**

```py
# try tf.name_scope(None) with tf.name_scope('clear'):  w1 = tf.Variable(1.0, dtype=tf.float32, name='w1')  with tf.name_scope(None):    w2 = tf.Variable(2.0, name='w2')print w1.nameprint w2.name
```

输出结果：

```py
clear/w1:0w2:0   # 已经清除了name scope 'clear'
```

**`variable_scope`的作用**

tensorflow 为了更好的管理变量,提供了variable scope机制。如何确定 get_variable 的 prefixed name ？

首先`variable scope`是可以嵌套的

```py
with tf.variable_scope('variable_scope_hello') as variable_scope:  vs1 = tf.get_variable('vs1', shape=[2,10], dtype=tf.float32)  print variable_scope   print type(variable_scope)  print variable_scope.name   print tf.get_variable_scope().original_name_scope   print vs1.name
  vs2 = tf.Variable([1,2], dtype=tf.int32, name='vs2')  print vs2.name  with tf.variable_scope('xxxx') as x_scope:
    vs3 = tf.get_variable('vs3', shape=[1,2], dtype=tf.float32)
    print vs3.name      # 输出结果：variable_scope_hello/xxxx/vs3:0    print tf.get_variable_scope().original_name_scope 
```

`get_varibale.name`以创建变量的 scope 作为名字的prefix



```py
def scope1():  with tf.variable_scope('scope1'):    var1 = tf.get_variable('var1', shape=[2], dtype=tf.float32)    print var1.name       # 'scope1/var1:0'    def scope2():      with tf.variable_scope('scope2'):        var2 = tf.get_variable('var2', shape=[2], dtype=tf.float32)      return var2     return scope2() res = scope1()print res.name   # 'scope1/scope2/var2:0'
```

再看一个例子，注意`return scope2()`的返回位置不同，`var2`变量的prefix亦不同。

```py
def scope1():  with tf.variable_scope('scope1'):    var1 = tf.get_variable('var1', shape=[2], dtype=tf.float32)    print var1.name       # 'scope1/var1:0'    def scope2():      with tf.variable_scope('scope2'):        var2 = tf.get_variable('var2', shape=[2], dtype=tf.float32)      return var2   return scope2() res = scope1()print res.name   # 'scope2/var2:0'  变量‘var2’不在‘scope1’中
```

注意`tf.variable_scope("name")`与`tf.variable_scope(scope)`的区别，看代码：

```py
with tf.variable_scope('scope'):  w = tf.get_variable('w', shape=[1])       print w.name        # name: 'scope/w'  with tf.variable_scope('scope'):    w = tf.get_variable('w', shape=[1])       print w.name      # name: 'scope/scope/w'with tf.variable_scope('scope'):  w1 = tf.get_variable('w1', shape=[1])     print w1.name       # name: 'scope/w1'  scope = tf.get_variable_scope()  with tf.variable_scope(scope):    w1 = tf.get_variable('w1', shape=[1])     print w1.name     # name: 'scope/w1'  因为两个变量的名字一样，会报错
ValueError: Variable scope/w1 already exists, disallowed. Did you mean to set reuse=True in VarScope?
```

**共享变量**

共享变量的前提是，变量的名字是一样的，变量的名字是由变量名和其scope前缀一起构成， `tf.get_variable_scope().reuse_variables()`是允许共享当前scope下的所有变量.

```py
with tf.variable_scope('scope1'):  w = tf.get_variable('w', shape=[1])       print w.name        # name: 'scope1/w'  with tf.variable_scope('scope2'):    w = tf.get_variable('w', shape=[1])       print w.name      # name: 'scope1/scope2/w'# 共享变量with tf.variable_scope('scope1', reuse=True):     w1 = tf.get_variable('w', shape=[1])    # 要求变量`scope/w`之前存在，在这里共享  print w1.name       # name: 'scope1/w'  print tf.get_variable_scope()
  print tf.get_variable_scope().reuse_variables()  with tf.variable_scope('scope2'):    w1 = tf.get_variable('w', shape=[1])     print w1.name     # name: 'scope1/scope2/w' 
```

**`name_scope`和`variable_scope`总结**

+ **使用`tf.Variable()`的时候，`tf.name_scope()`和`tf.variable_scope()`都会给 Variable 和 op 的 name属性加上前缀**。 
+ **使用`tf.get_variable()`的时候，`tf.name_scope()`就不会给 `tf.get_variable()`创建出来的Variable加前缀**。


### TensorFlow IO

报错:

```
OutOfRangeError (see above for traceback): RandomShuffleQueue '_2_input/shuffle_batch/random_shuffle_queue' is closed and has insufficient elements (requested 1, current size 0)	 [[Node: input/shuffle_batch = QueueDequeueManyV2[component_types=[DT_STRING], timeout_ms=-1, _device="/job:localhost/replica:0/task:0/cpu:0"](input/shuffle_batch/random_shuffle_queue, input/shuffle_batch/n)]]
#  requested 1. 其中1为batch_size
```

源于下面代码：

```py
filename_queue = tf.train.string_input_producer(file_name, num_epochs=max_epoch, seed=seed)
```

改为下面就worker

```py
filename_queue = tf.train.string_input_producer(file_name, seed=seed)
```

原因在于num_epochs参数，具体未知。

### 错误：

```
ValueError: initial_value must have a shape specified: Tensor("input/ParseExample/ParseExample:6", shape=(?,), dtype=float32)
```

```
TypeError: Expected binary or unicode string, got 0
代码：
``` 

<h2 id="2.TensorFlow Embedding">2. Embeddings</h2>

<h3 id="2.1.embedding_lookup">2.1. embedding_lookup</h3>

先看一个示例：

```py
#!/usr/bin/env python#-*- coding: utf-8 -*-import tensorflow as tf import numpy as npinput_ids = tf.placeholder(dtype=tf.int32, shape=[None])embedding = tf.Variable(np.identity(5, dtype=np.int32))input_embedding = tf.nn.embedding_lookup(embedding, input_ids)init = tf.global_variables_initializer()with tf.Session() as sess:  sess.run(init)  print("embedding.eval:\n{}".format(embedding.eval()))  result = sess.run(input_embedding, feed_dict = {input_ids: [1,2,3,0,3,2,1]})  print ("input_embedding: \n{}".format(result))
```
输出为：

```py
embedding.eval:[[1 0 0 0 0] [0 1 0 0 0] [0 0 1 0 0] [0 0 0 1 0] [0 0 0 0 1]]input_embedding: [[0 1 0 0 0] [0 0 1 0 0] [0 0 0 1 0] [1 0 0 0 0] [0 0 0 1 0] [0 0 1 0 0] [0 1 0 0 0]]
```

如果把第8行的embedding替换为下面形式：

```pyembedding = tf.Variable(np.asarray([[0.11,0.12,0.13], [0.21,0.22,0.23], [0.31,0.32,0.33], [0.41,0.42,0.43], [0.51,0.52,0.53]]))
```
运行结果为：

```bash
embedding.eval:[[ 0.11  0.12  0.13] [ 0.21  0.22  0.23] [ 0.31  0.32  0.33] [ 0.41  0.42  0.43] [ 0.51  0.52  0.53]]input_embedding: [[ 0.21  0.22  0.23] [ 0.31  0.32  0.33] [ 0.41  0.42  0.43] [ 0.11  0.12  0.13] [ 0.41  0.42  0.43] [ 0.31  0.32  0.33] [ 0.21  0.22  0.23]]
```

解读上述代码：先使用palceholder定义了一个未知变量`input_ids`用于存储索引，和一个已知变量embedding（这里是一个5*5的对角矩阵，方法：`tf.Identity()`）。

函数原型为：

```py
embedding_lookup(
    params,
    ids,
    partition_strategy='mod',
    name=None,
    validate_indices=True,
    max_norm=None
)
```

params参数是一些tensor组成的列表或者单个的tensor。ids一个整型的tensor，每个元素将代表要在params中取的每个元素的第0维的逻辑index，这个逻辑index是由`partition_strategy`来指定的。

`embedding_lookup`函数的本质：**只是想做一次常规的线性变换而已，Z = WX + b。**  `embedding_lookup`不是简单的查表，id对应的向量是可以训练的，训练参数个数应该是
 $category\_num \cdot embedding\_size$，也就是说lookup是一种全连接层。

由于输入都是One-Hot Encoding（稀疏编码的结果），和矩阵相乘相当于是取出Weights矩阵（对应`embedding_lookup`参数第一个参数`params`）中对应的那一行，所以tensoflow封装了方法`tf.nn.embedding_lookup(params, ids)`接口，更加方便的表示意思。查找params对应的ids行。等于说变相的进行了一次矩阵相乘运算，其实就是一次线性变换。

参数params即为lookup table，根据`ids`对应的行，然后执行对应的操作（参数`partition_strategy`，可以取值为`sum`／`mean`／`sqrtn`）

`embedding_lookup`函数总结：

1. 适用场景：`embedding_lookup`适用于**做（稀疏）id类特征的embedding表示**（由google的deep&wide提出，但隐藏了具体实现细节）
2. `params`为要查找的信息（即lookup table，稠密表示），ids为要查找的行集合。

> [7.6 DNN在搜索场景中的应用(作者：仁重)](http://www.cnblogs.com/hujiapeng/p/6236857.html) 中介绍了关于user/query/item之间的编码方式与网络构建过程，值得参考。


<h3 id="2.1.embedding_lookup_sparse">2.1. embedding_lookup_sparse</h3>


[`embedding_lookup_sparse`](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup_sparse)函数原型：

示例：

```py
#!/usr/bin/env python#-*- coding: utf-8 -*-import tensorflow as tfimport numpy as npdef test_embedding_lookup_sparse():  a = np.arange(12).reshape(3,4)  b = np.arange(8, 20).reshape(3, 4)  c = np.arange(20, 32).reshape(3, 4)  print("a:\n{}\nb:\n{}\nc:\n{}".format(a, b, c))   a = tf.Variable(a, dtype = tf.float32)  b = tf.Variable(b, dtype = tf.float32)  c = tf.Variable(c, dtype = tf.float32)  sp_ids = tf.SparseTensor(indices = [[0,0], [0,1], [1,0], [1,2]], values = [1,2,2,0], dense_shape = (2,3))  sp_weights = tf.SparseTensor(indices = [[0,0], [0,1], [1,0], [1,2]], values = [1.1,1.1,1.1,1.1], dense_shape = (2,3))  sp_ids_weights_result = tf.nn.embedding_lookup_sparse(a, sp_ids, sp_weights, combiner = "sum")    ids = tf.SparseTensor(indices=[[0,0], [0,2], [1,0], [1,1]], values = [1,2,2,0], dense_shape = (2,3))  sp_ids_result = tf.nn.embedding_lookup_sparse((a,b,c), ids, None, combiner = "sum")  init = tf.global_variables_initializer()  with tf.Session() as sess:    sess.run(init)    print("sp_ids_result: \n{}".format(sess.run(sp_ids_result)))     print("sp_ids_weights_result: \n{}".format(sess.run(sp_ids_weights_result)))def main():  test_embedding_lookup_sparse()if __name__ == '__main__':  main()
```

输出结果为：

```py
a:[[ 0  1  2  3] [ 4  5  6  7] [ 8  9 10 11]]b:[[ 8  9 10 11] [12 13 14 15] [16 17 18 19]]c:[[20 21 22 23] [24 25 26 27] [28 29 30 31]]
sp_ids_result: [[[ 28.  30.  32.  34.]  [ 36.  38.  40.  42.]  [ 44.  46.  48.  50.]] [[ 20.  22.  24.  26.]  [ 28.  30.  32.  34.]  [ 36.  38.  40.  42.]]]sp_ids_weights_result:  (params * sp_weights) 然后执行combiner操作[[ 13.20000076  15.40000057  17.60000038  19.80000114] [  8.80000019  11.00000095  13.19999981  15.40000057]]
```

> 解释：
> 1. `sp_ids = tf.SparseTensor(indices = [[0,0], [0,1], [1,0], [1,2]], values = [1,2,2,0], dense_shape = (2,3))` indices表示稀疏索引，values里的值为参数params里的下标。
> 2. `sp_ids_result = tf.nn.embedding_lookup_sparse((a,b,c), ids, None, combiner = "sum")` 这里`sp_weights=None`使用默认值（weight=1）
> 3. `sp_weights = tf.SparseTensor(indices = [[0,0], [0,1], [1,0], [1,2]], values = [1.1,1.1,1.1,1.1], dense_shape = (2,3))`制定稀疏weight的值。

```py
embedding_lookup_sparse(
    params,
    sp_ids,
    sp_weights,
    partition_strategy='mod',
    name=None,
    combiner=None,
    max_norm=None
)
```

params参数与`embedding_lookup`函数一致，`sp_ids`和`sp_weights`为`SparseTensor`。

tensorflow针对稀疏数据（`SparseTensor`）做embedding。的函数是[`embedding_lookup_sparse`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/embedding_ops.py#L108)

**用`embedding_lookup_sparse`应用于稀疏LR模型**

```py
def test_embedding_lookup_sparse_for_lr_model():  weight_size = 1000000  weights = np.arange(weight_size)  print("weights[:5]: {}".format(weights[:5]))  weights = tf.Variable(weights, dtype=tf.float32)  instance_indices = [[0,0], [0,1], [0,2], [1,0]]  feature_ids = [1,2,100,100]  feature_values = [0.01, 0.01, 0.01, 0.2]  instance_shape = (2,3)  sp_ids = tf.SparseTensor(indices = instance_indices, values = feature_ids, dense_shape = instance_shape)  sp_values = tf.SparseTensor(indices = instance_indices, values = feature_values, dense_shape = instance_shape)   wTx = tf.nn.embedding_lookup_sparse(weights, sp_ids, sp_values, combiner = "sum")  init = tf.global_variables_initializer()  with tf.Session() as sess:    sess.run(init)    result = sess.run(wTx)    print("wTx: {}, shape: {}".format(result, sess.run(tf.shape(wTx))))
```

--
## 第二部分：TensorFlow Serving

TensorFlow Serving官网地址：https://www.tensorflow.org/serving/

# Serving与生产环境 问题列表 

| id | issue | solution |
| --- | --- | --- |
| 1 | TF Serving模型自动更新的逻辑是？| 
| 2 | 线上服务支持的接口与调用方式，接口是否灵活(多业务、slot场景)？ | 
| 3 | 线上服务要求长期稳定，资源非抢占，是否满足？|
| 4 | 怎么支持ABtest? 启动两个serving 服务名称是否冲突？|
| 5 | 高并发请求一些问题：负载均衡，线程池 | 
| 6 | Serving线上监控如何看？ 预测失败如何处理？失败比例？ | 
| 7 | 参数：最大内存上限？cores能提高单词预测的性能么？|
| 8 | `<appkey, model_name, path>`是否都会对应一个serving任务？如此会启动很多serving |
| 9 | hdfs模型产出路径包含日期，是否支持？|

<h2 id="1.TensorFlow Serving入门">1.TensorFlow Serving入门</h2>

<h3 id="1.1. 安装与使用">1.1.安装与使用</h3>

**安装依赖：`jdk1.18`, [`gRPC`](https://github.com/grpc/grpc/tree/master/src/python/grpcio)**

```bash
# 安装jdk1.8
# 安装bazel (依赖jdk1.8)
which bazel
# 安装grpc
sudo pip install grpcio
```
**安装tensorflow serving(源码安装)**

```sh
# 下载
git clone --recurse-submodules https://github.com/tensorflow/serving
cd serving 
cd tensorflow && ./configure
cd ..
bazel build -c opt tensorflow_serving/...    # 编译tf_serving所有源码
# 编译结果保存在：`bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server`
# 测试编译结果
bazel test -c opt tensorflow_serving/...
```

**安装遇到的错误**

+ [`"virtual memory exhausted: Cannot allocate memory"`](http://muchfly.iteye.com/blog/2296506)

表示虚拟内存耗尽，需要增加swap大小。操作如下：

```sh
#  查看当前swap大小
> free -m
              total        used        free      shared  buff/cache   availableMem:           3691         534        2992           0         164        2942Swap:          1021         820         201
# 创建swap文件（目录可以自己指定，这里指定为/var/swap）
> sudo dd if=/dev/zero of=/var/swap bs=2048 count=2048000
2048000+0 records in2048000+0 records out4194304000 bytes (4.2 GB, 3.9 GiB) copied, 4.87359 s, 861 MB/s# 建立swap
> sudo mkswap /var/swap
Setting up swapspace version 1, size = 3.9 GiB (4194299904 bytes)no label, UUID=a378f3a3-f16f-4294-afa3-e4ff31171678
# 启动swap
> sudo swapon /var/swap
swapon: /var/swap: insecure permissions 0644, 0600 suggested.# 再次查看swap大小
> free -m
              total        used        free      shared  buff/cache   availableMem:           3691         550         135           0        3006        2857Swap:          5021         798        4223```

<h3 id="1.2. TFServing单机测试">1.2.TFServing单机测试</h3>

以TFServing官方提供的MNIST为例，

客户端调用model错误

+ `tensorflow.python.framework.errors_impl.NotFoundError:  .... _prefetching_ops.so: undefined symbol: _ZNK6google8protobuf5Arena17OnArenaAllocationEPKSt9type_infom`


--
TensorFlow错误总结

1. [`tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value Variable`](http://blog.csdn.net/ztf312/article/details/72859852)
    错误原因：没有初始化变量，**Tensorflow中，所有变量都必须初始化才能使用**。
    解决方案：**添加`sess.run(tf.initialize_all_variables())`**

2. [`TypeError: unhashable type: 'numpy.ndarray'`](https://stackoverflow.com/questions/43664985/typeerror-unhashable-type-numpy-ndarray-tensorflow)

    错误原因：placeholder名称与其它变量有重复，注意变量的区分即可。


$
1. 离线数据转化： \longrightarrow \text{tf.train.Example(TFRecord格式)} \\\ 
2. AI模型数据加载与解析. 
$

通过feature key的配置来实现，离线数据转化和训练时输入解析数据一致性。
