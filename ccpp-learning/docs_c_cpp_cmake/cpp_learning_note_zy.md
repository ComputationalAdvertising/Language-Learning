## CPlusPlus Learning Note

+ author: zhouyongsdzh@foxmail.com
+ date: 20160829

### 目录

--
#### 基础知识
+ namespace
+ io系统
+ struct


#### 数据结构与算法
+ string
+ vector
+ sort
+ math

#### C++特性
+ class与继承
+ virtual虚函数
+ 智能指针

#### C++11新特性
+ std::functional
+ std::bind
+ 匿名函数（lambda表达式）

#### 那些坑儿

+ **```undefined reference to `...` ```**


---
### 基础知识

#### namespace 使用

命名空间(namespace)是一种描述逻辑分组的机制，可以将按某些标准在逻辑上属于同一个任务中的所有类声明放在同一个命名空间中。标准C++库（不包括标准C库）中所包含的所有内容（包括常量、变量、结构、类和函数等）都被定义在命名空 间std（standard标准）中了。

+ 定义命名空间

有两种形式的命名空间——有名的和无名的。命名空间的定义格式为：

```
namespace 命名空间名 {	// 有名
	// 声明序列可选，可包含常量、变量、结构、类、函数等;
}
---
namespace {				//无名
	// 声明序列可选，可包含常量、变量、结构、类、函数等;
}
```

+ 内部定义与外部定义

命名空间的成员，是在命名空间定义中的花括号内声明了名称。可以在命名空间的定义内，定义命名空间 的成员（内部定义）。也可以只在命名空间的定义内声明成员，而在命名空间的定义之外，定义命名空间的成员（外部定义）。 举例：

```
out.h
// 定义外部命名空间：outer
namespace outer {
       	int i;
       	namespace inner {      	// 定义子命名空间：inner。⚠️：字命名空间只能定义 不能声明。
       		void f() {     		// inner成员f的定义，其中的i为outer::i. 注意先后顺序。
       			i++;   		// outer::i
       		}
       		int i; 			// 子命名空间成员i
       		void g() {
       			i++;   		// 因为定义在子命名空间i之后，所以是inner::i
       		}
       		void h();      		// 可以声明成员
       	}
       	void f();      		// outer f declare
       	//namespace inner2;    	// 错误，不能声明子空间
}

void outer::f() {i--;} 		// 命名空间outer的f外部定义
void outer::inner::h() {i--; } 	// 命名空间inner成员h的外部定义
// namespace outer::inner2{}   	// 错误，不能在外部定义子命名空间。

namespace outer {
       	//std::string str = "I love this world!!!";
       	std::string str;
}
```

> **⚠️：子命名空间 在命名空间中只能定义，不能声明。**

也不能直接使用“命名空间名::成员名 ……”定义方式，为命名空间添加新成员，而必须先在命名空间的定义中添加新成员的声明。另外，命名空间是开放的，即可以随时把新的成员名称加入到已有的命 名空间之中去。方法是，多次声明和 定义同一命名空间，每次添加自己的新成员和名称。例如：

```
namespace A {
	int i;
	void f();
}		// A中含有i和f()两个成员
---
namespace A {
	std::string str;
	void g();
}		// A中含有i,f(),str,g()四个成员
```

+  namespace示例. out.cpp内容：

```
#include "out.h"
#include "out_cp.h"
#include <iostream>
using namespace outer::inner;
using namespace std;

int main(int argc, char* argv[])
{
       	outer::str = "I love you!!!1";
       	std::cout << "outer::str: " << outer::str << std::endl;
       	outer::i = 0;
       	outer::f();
       	std::cout << "[1]. i: " <<  outer::i << std::endl;
       	f();   		// inner::f()
       	std::cout << "[2]. i: " <<  outer::i << std::endl;
       	i = 1;
       	g();   		// i=2
       	std::cout << "[3]. i: " <<  i << std::endl;
       	h();   		// i = 1
       	std::cout << "[4]. i: " <<  i << std::endl;
       	return 0;
}

```

--
## C++特性

--
### [智能指针](http://blog.csdn.net/xt_xiaotian/article/details/5714477)

为什么要存在智能指针？为什么要使用智能指针？博客[智能指针的作用与原理](http://blog.csdn.net/chen825919148/article/details/8155411)已经讲解的很清楚。

在C++中创建一个对象时（堆对象），需要保证**对象一定会被释放，但只能释放一次，并且释放后指向该对象的指针应该马上归0。**

> 一定要有delete对象，但不可重复delete（会释放别人的空间），delete后对应的指针如果不归0（或null），该指针可能会操作其对应的空间，侵犯别人的地盘（指针霸气侧漏）。
> 
> 以上程序员操作不当，都会成为导致内存泄漏的主要原因。

我们需要一个类似Java能自动回收内存的机制。因此智能指针诞生，**智能指针是借鉴了Java内存回收机制（引用计数器机制）**，从而不用由程序员执行delete/free操作了。

> [知乎上对智能指针的回答](http://www.zhihu.com/question/20368881) 有几个比较精彩：

**智能指针的作用**

1. 防止用户忘记调用delete；
2. 异常安全（在try/catch代码段里，即使写入了delete，也可能发生异常）；
3. **把value语义转化为reference语义**；

> C++和Java有一处很大的区别在于**语义不同**，java代码段：
> 
```
Animal a = new Animal(); Animal b = a;
```
> 这里其实只生成了一个对象，a和b仅仅是把持对象的引用而已。但在C++中不是这样：
> ```Animal a; 	Animal b；``` 确实是生成了两个对象。
> 
> Java中往容器中放对象，实际放入的是引用，不是真正的对象。而在C++的vector中push_back采用的是值拷贝。如果像实现Java中的引用语义，此时可使用智能指针。

 
**智能指针与普通指针的区别**

智能指针实际上是对普通指针加了一层封装机制，这层封装机制的目的是为了使得智能指针可以方便的管理一个对象的生命周期。
> 在智能指针中，一个对象什么时候在什么条件下被析构（reset）或者删除由智能指针本身决定，用户不需要管理。

另外一个区别在于，智能指针本质上是一个对象，行为表现的像一个指针。

下面先介绍智能指针的种类、区别与联系。

示例类：

```
class Base {
```

#### [```auto_ptr```](http://www.cplusplus.com/reference/memory/auto_ptr/)

首先要说明的是，在C++11中，已经废除了```template <class X> class auto_ptr```这样一个类模型，取而代之的是```std::unique_ptr```：与```auto_ptr```功能类似，但是提升了安全性，添加了deleter功能，并且支持数组。

> 如果在C++11中使用```std::auto_ptr```会提示下面类似的warning：
> 
```
warning: ‘template<class> class std::auto_ptr’ is deprecated [-Wdeprecated
-declarations]
```

```auto_ptr```提供的成员函数主要有：get(), operator* , operator-> , operator= , release(), reset().

示例：

```
int main(int argc, char * agrv[]) {
```
**std::auto_ptr几点说明：**

1. auto_ptr对象赋值，会失去内存管理所有权，使得```my_auto_ptr```悬空，最后使用时会导致程序崩溃。因此，**使用```std::auto_ptr```时，尽量不要使用````operator=```操作符**，如果使用了，不要再使用先前对象。
2. 使用```std::auto_ptr release()```函数时，会发现创建出来的对象没有被析构，导致内存泄漏。这是因为```release```函数不会释放对象，仅仅归还所有权。释放对象可使用```reset()```函数。
3. ```std::auto_ptr```不能当作参数传递，同时其管理的对象也不能放入```std::vector```等容器中，因为```operator=```问题。

+ 参考链接：http://blog.csdn.net/xt_xiaotian/article/details/5714477


#### [```unique_ptr```](http://www.cplusplus.com/reference/memory/unique_ptr/?kw=unique_ptr)

上面的```std::auto_ptr```在复制构造或者```operator=```时，原先的对象就报废了，因为所有权转移到新对象去了，是程序崩溃的隐患。为了避免此现象发生，```std::unique_ptr```很好的解决了这个问题，其**不提供智能指针的```=```操作，必须通过间接或显示的交出所有权（```std::move```, since c++11）**。

```
std::unique_ptr<Base> fetch_unique_ptr() {

int main(int argc, char * agrv[]) {
	/* std::unqiue_ptr */
	// my_unique_ptr5->PrintInfo();		// error
```

**std::unique_ptr几点说明：**

1. 无法进行复制构造和赋值操作：意味着无法得到指向同一个对象的两个unique_ptr. 但提供了移动构造赋值和显示赋值功能。
2. 为动态申请的内存提供异常安全；可以将动态申请内存的所有权传递给某个函数；从某个函数返回动态申请内存的所有权；
3. 可以作为容器元素。

```std::auto_ptr```和```std::unique_ptr```都是某一块内存独享所有权的智能指针。实际应用中会出现**某一块内存允许多个智能指针共享**，该当如何？下面的```std::shared_ptr```可以满足这个场景。

> 分布式机器学习评估模型指标，即计算auc, logloss等指标时，可以使用```std::shared_ptr```

#### [```shared_ptr```](http://www.cplusplus.com/reference/memory/shared_ptr/?kw=shared_ptr)


--
## C++11新特性

--
### std::functional

--
### std::bind

--
### [匿名函数（lambda表达式）](http://blog.csdn.net/augusdi/article/details/11773943)

Lambda表达式具体形式：

**```[capture](paramters) -> return-type {body}```**

如果没有参数，空的圆括号```()```可以省略。如果函数体只有一条return语句或者返回类型为void时，返回值也可以忽略。形如：

**```[capture](parameters) {body}```**

几个Lambda函数的例子：

```
[](int x, int y) { return x+y; }			// 隐式返回类型
[](int& x) { ++x; }			// 没有return语句，则lambda函数的返回值类型是'void'
[]() { ++global_x; }		// 没有参数，仅访问某个全局变量
```

可以像下面显示指定返回类型：

```[](int x, int y) -> int { int z=x+y; return z; }```

什么也不返回的Lambda函数可以省略返回类型，而不需要使用```-> void```形式。

**Lambda函数可以引用在它之外声明的变量**, 这些变量的集合叫做一个**闭包**。闭包被定义在Lambda表达式声明中的方括号```[]```内。这个机制允许这些变量被按值或按引用捕获。示例：

```
[]			// 未定义变量。试图在Lambda内使用任何外部变量都是错误的
[x, &y]		// x按值捕获，y按引用捕获
[&]			// 用到的任何外部变量都隐式按引用捕获
[=]			// 用到的任何外部变量都隐式按值捕获
[&, x]		// x显示地按值捕获。其它变量按引用捕获
[=, &z]		// z按引用捕获，其它变量按值捕获
```

示例说明Lambda表达式用法:

```
std::vector<int> list;
int total = 0;
for (int i = 0; i < 5; ++i) list.push_back(i);
std::for_each(begin(list), end(list), [&total](int x) { 
	total += x;
});			// 计算list中所有元素的总和。
```
变量total被存为lambda函数闭包的一部分。因为total是栈变量（局部变量）total的引用，所以可以改变它的值。

Lambda函数是一个依赖于实现的函数对象类型，这个类型的名字只有编译器知道。**如果用户想把Lambda函数做一个参数来传递，那么行参的类型必须是模版类型或者必须能创建一个```std::function```类似的对象去捕获lambda函数**。使用auto关键字可以帮助存储lambda函数。

```
auto my_lambda_func = [&](int x) { /* ... */ };
auto my_onheap_lambda_func = new auto([=](int x) { /* ... */ });
```

下面例子把匿名函数存储在变量、数组或vector中，并把它们当作命名参数来传递：

```
#include <functional>
#include <vector>
#include <iostream>
using namespace std;
 
double eval(std::function<double(double)> f, double x = 2.0) {
	return f(x);
}

int main(int argc, char* argv[])
{
	// use std::function capture lambda function
	std::function<double(double)>	f0		= [](double x) { return 1; };
	// use auto keywords save lambda function
	auto							f1		= [](double x) { return x; };
	decltype(f0)					fa[3]	= {f0, f1, [](double x) {return x*x;}};
	std::vector<decltype(f0)>		fv		= {f0, f1};
	fv.push_back					([](double x) {return x*x; });

	std::cout << "test-0:\n";
	for (auto i = 0; i < fv.size(); ++i)	std::cout << fv[i](2.0) << "\n";
	std::cout << "test-1:\n";
	for (auto i = 0; i < 3; ++i)			std::cout << fa[i](2.0)	<< "\n";
	std::cout << "test-2:\n";
	for (auto &f: fv)						std::cout << f(2.0)	<< "\n";
	std::cout << "test-3:\n";
	for (auto &f: fa)						std::cout << f(2.0)	<< "\n";
	std::cout << eval(f0) << "\n";
	std::cout << eval(f1) << std::endl;
	return 0;

}
```

## 那些坑儿

--
### [```undefined reference to `...` ```](http://blog.csdn.net/jfkidear/article/details/8276203)

异常示例：```undefined reference to `dmlc::Config::Config(std::istream&, bool)' ```

主要原因是```*.cc```程序没有**链接**dmlc库函数，需要在```*.cc```对应```CMakeLists.txt```文件添加 **链接库函数代码，即```target_link_libraries(${exec_name} dmlc)```**，相当于在g++上添加了参数```-ldmlc```

> 类似的问题： ```undefined reference to 'pthread_create'``` 需要添加```-lpthread```




