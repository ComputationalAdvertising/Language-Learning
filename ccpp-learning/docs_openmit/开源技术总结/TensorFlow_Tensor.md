## TensorFlow Tensor


Tensor是所有深度学习框架（TF、MxNet等）最核心的组件。在TensorFlow中，将数据表示为张量（Tensor）。

**`Tensor是一个多维数组。标量：0阶张量；向量：1阶张量；矩阵：2阶张量；更高维：n阶丈量`**


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
