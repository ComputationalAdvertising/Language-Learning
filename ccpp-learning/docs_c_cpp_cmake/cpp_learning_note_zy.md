## CPlusPlus Learning Note

+ author: zhouyongsdzh@foxmail.com
+ date: 20160829


issue:

2. static_cast/dynamic_cast使用场景
3. &, * 在参数调用中的使用场景

### 目录
--

#### [1. 基础知识](#1.基础知识)

+ [1.1. C++关键字](#1.1.C++关键字)

    |[[namespace](#1.1.1.namespace)]|[[typedef](#1.1.2.typedef)]|[[template](#1.1.3.template)]| [[explicit](#1.1.4.explicit)]
    | --- | --- | --- | --- |
    
+ [1.2. IO系统](#1.2.IO系统) 

#### [2. 数据结构与算法](#2.数据结构与算法)

| [[string](#2.1.string)]| [[vector](#2.2.vector)]| [[iterator](#2.3.iterator)] | [[map](#2.4.map)] | [[struct](#2.5.struct)] | [[math](#2.6.math)] | [[sort](#2.7.sort)] | [[random](#2.8.random)]
| --- | --- | --- | --- | --- | --- | --- | --- |

#### [3. C++特性](#3.C++特性)

| [[class]()] | [[virtual]()] | [[smart_ptr](#3.3.smart_ptr)] |
| --- | --- | --- |

#### 4. C++11新特性

| [[std::functional]()] | [[std::bind]()] | [[lambda]()]
| --- | --- | --- |

#### 5. 那些坑儿

+ **```undefined reference to `...` ```**


<h2 id="1.基础知识">1. 基础知识</h2> 


<h3 id="1.1.C++关键字">1.1. C++关键字</h3> 


<h4 id="1.1.1.namespace">1.1.1. namespace</h4> 


命名空间(namespace)是一种描述逻辑分组的机制，可以将按某些标准在逻辑上属于同一个任务中的所有类声明放在同一个命名空间中。标准C++库（不包括标准C库）中所包含的所有内容（包括常量、变量、结构、类和函数等）都被定义在命名空 间std（standard标准）中了。

+ 定义命名空间

有两种形式的命名空间——有名的和无名的。命名空间的定义格式为：

```c++
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

```c++
out.h
// 定义外部命名空间：outer
namespace outer {
       	int i;
       	namespace inner {      	// 定义子命名空间：inner。⚠️：子命名空间只能定义 不能声明。
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

```c++
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

```c++
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

<h4 id="1.1.4.explicit">1.1.4. explicit</h4>

C++中的explicit关键字只能用于**修饰只有一个参数的类构造函数**, 它的作用是表明该构造函数是显示的, 而非隐式的；与之对应的关键字是**implicit**.

显示声明的构造函数和隐式声明的有何区别呢？先看一个隐式声明的例子（未使用explicit即默认为隐式声明）：

```c++
#include <malloc.h>#include <cstring>#include <iostream>using namespace std;
class NonExplicit {  public:    char * pstr_;    int size_;    NonExplicit(int size) {      std::cout << "NonExplicit pre-malloc " << size << " byte space." << std::endl;      size_ = size;      pstr_ = (char *)malloc(size_ + 1);      memset(pstr_, 0, size + 1);    }    NonExplicit(const char * ch) {      std::cout << "NonExplicit " << ch << ", strlen:" << strlen(ch) << std::endl;      int size = strlen(ch);      pstr_ = (char *)malloc(size + 1);      strcpy(pstr_, ch);      size_ = strlen(pstr_);    }    ~NonExplicit() {       std::cout << "~NonExplicit. pstr_: " << pstr_ << std::endl;      //free(pstr_);  // may be issue:  double free or corruption (fasttop);    }};int main(int argc, char * argv[]) {  // ---- non explicit ----  NonExplicit ne1(24);      // ok  NonExplicit ne2 = 10;     // ok  //NonExplicit ne3;          // failure. not found constructor function matched  NonExplicit ne4("aaaa");  // ok  NonExplicit ne5 = "bbbb"; // ok  NonExplicit ne6 = 'a';    // ok, convert to 'a' ascii code.  ne4 = "ccccccccccc";  ne5 = "xxxxxxx";}
```

代码中，`NonExplicit ne2 = 10;`为什么可以呢？因为在c++中，如果构造函数只有一个参数时，那么在编译的时候会有一**个缺省的转换操作**：将该构造函数对应数据类型的数据转化为该类对象。这里，编译器自动将整型转换为NonExplicit类对象，即`NonExplicit ne2(10);`

`NonExplicit ne2 = 10;`和`NonExplicit ne6 = 'a';初始化类显得不伦不类，容易让人疑惑，有什么办法可以阻止这种用法呢？答案就是使用`explicit`关键字。把上面的代码修改如下：

> **注意⚠**：在拷贝构造函数或者拷贝赋值运算符实现时，若不注意，容易出现`double free or corruption (fasttop)`的问题；

```c++
#include <malloc.h>#include <cstring>#include <iostream>using namespace std;
class Explicit {  public:    char * pstr_;    int size_;    explicit Explicit(int size) {		// 使用关键字explicit的类声明, 显示转换       std::cout << "Explicit pre-malloc " << size << " byte space." << std::endl;      size_ = size;      pstr_ = (char *)malloc(size_ + 1);      memset(pstr_, 0, size + 1);    }    Explicit(const char * ch) {      std::cout << "Explicit " << ch << ", strlen:" << strlen(ch) << std::endl;      int size = strlen(ch);      pstr_ = (char *)malloc(size + 1);      strcpy(pstr_, ch);      size_ = strlen(pstr_);    }    ~Explicit() {       std::cout << "~Explicit. pstr_: " << pstr_ << std::endl;      //free(pstr_);  // may be issue:  double free or corruption (fasttop);    }};int main(int argc, char * argv[]) {  // ---- explicit ----  Explicit e1(24);      // ok  //Explicit e2 = 10;     // no. explicit取消了隐式转换  //NonExplicit e3;          // failure. not found constructor function matched  Explicit e4("aaaa");  // ok  //Explicit e5 = "bbbb"; // no. must be explicit using constructor  //Explicit e6 = 'a';    // no. must be explicit using constructor
  e3 = e1;				  // no. 因为取消了隐式转换，除非类实现操作符"="的重载}
```

**explicit关键字的作用就是防止类构造函数的隐式自动转换**. 

explicit关键字只对有一个参数的类构造函数有效, 如果类构造函数参数大于或等于两个时, 是不会产生隐式转换的, 所以explicit关键字也就无效了。

但是, 也有一个例外, 就是当除了第一个参数以外的其他参数都有默认值的时候, explicit关键字依然有效, 此时, 当调用构造函数时只传入一个参数, 等效于只有一个参数的类构造函数。

### io系统

<h2 id="2.数据结构与算法">2. 数据结构与算法</h2>

<h3 id="2.2.vector">2.2. vector</h3>

### struct结构体

### random随机数

http://blog.csdn.net/x356982611/article/details/50909142

C++11引入的伪随机数发生器，随机数抽象成生成器和分布器两部分。生成器用来产生随机数，分布器用来生成特征分布的随机数。示例：

```c++
std::random_device rd;		// 生成一个随机数作为种子
std::uniform_int_distribution<int> uni_dist(0, 99999);  // 指定范围的随机数发生器
std::cout << uni_dist(rd) << std::endl;
```

<h2 id="3.C++特性">3. C++特性</h2>


参考[智能指针](http://blog.csdn.net/xt_xiaotian/article/details/5714477)

<h4 id="3.3.smart_ptr">3.3. smart_ptr</h4> 



为什么要存在智能指针？为什么要使用智能指针？博客[智能指针的作用与原理](http://blog.csdn.net/chen825919148/article/details/8155411)已经讲解的很清楚。

在C++中创建一个对象时（堆对象），需要保证**对象一定会被释放，但只能释放一次，并且释放后指向该对象的指针应该马上归0。**

> 1. 内存泄漏：一定要有delete对象；
> 2. 重复释放：但不可重复delete（会释放别人的空间）；
> 3. 野指针：delete后对应的指针如果不归0（或null），该指针可能会操作其对应的空间，侵犯别人的地盘（指针霸气侧漏）。
> 
> 以上指针操作不当的行为，都会成为导致内存泄漏的主要原因。

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
> 这里其实只生成了一个对象，a和b仅仅是共同把持一个对象的引用而已。但在C++中不是这样：
> ```Animal a; 	Animal b；``` 确实是生成了两个对象。
> 
> Java中往容器中放对象，实际放入的是引用，不是真正的对象。而在C++的vector中push_back采用的是值拷贝。如果像实现Java中的引用语义，此时可使用智能指针。

 
**智能指针与普通指针的区别**

智能指针实际上是对普通指针加了一层封装机制，这层封装机制的目的是为了使得智能指针可以方便的管理一个对象的生命周期。
> 在智能指针中，一个对象什么时候在什么条件下被析构（reset）或者删除由智能指针本身决定，用户不需要管理。

另外一个区别在于，智能指针本质上是一个对象，行为表现的像一个指针。

下面先介绍智能指针的种类、区别与联系。

示例类：

```c++
class Base {	public:		Base(int param = 0) {			number_ = param;			std::cout << "Create Base Object. number_: " << number_ << std::endl;		}		~Base() { std::cout << "~Base: " << number_ << std::endl; }		void PrintInfo() { std::cout << "PrintInfo: " << info.c_str() << std::endl; }		std::string info;		int number_;};class Derived : public Base {	public:		Derived(int param) { std::cout << "Create Derived" << std::endl; }		~Derived() { std::cout << "~Derived" << std::endl; }};
```

#### [```auto_ptr```](http://www.cplusplus.com/reference/memory/auto_ptr/)

首先要说明的是，在C++11中，已经废除了```template <class X> class auto_ptr```这样一个类模型，取而代之的是```std::unique_ptr```。与```auto_ptr```功能类似，但是提升了安全性，添加了deleter和"operator="功能，并且支持数组。

> 如果在C++11中使用```std::auto_ptr```会提示下面类似的warning：
> 
```
warning: ‘template<class> class std::auto_ptr’ is deprecated [-Wdeprecated
-declarations]  std::auto_ptr<Base> auto_p(Base);       ^In file included from /usr/include/c++/5/memory:81:0,/usr/include/c++/5/bits/unique_ptr.h:49:28: note: declared here   template<typename> class auto_ptr;
```

```auto_ptr```提供的成员函数主要有：get(), operator* , operator-> , operator= , release(), reset().

示例：

```c++
int main(int argc, char * agrv[]) {	/* std::auto_ptr */	std::auto_ptr<Base> my_auto_ptr(new Base(11));	// create object. output: 11	if (my_auto_ptr.get()) {					// judge smart_ptr is null		my_auto_ptr->PrintInfo();				// use operator-> to call function of ptr-object		my_auto_ptr->info = "hello auto_ptr";	// use operator= to assign value		my_auto_ptr->PrintInfo();		(*my_auto_ptr).info += "!!!!!";			// use operator* return inner object, then use '.' to call func		my_auto_ptr->PrintInfo();	} else {		std::cout << "my_auto_ptr is null" << std::endl;	}	// [expirment] auto_ptr: object copy	if (my_auto_ptr.get()) {		std::auto_ptr<Base> my_auto_ptr2(new Base(22));		my_auto_ptr2->info = "this world";		my_auto_ptr2 = my_auto_ptr;				// copy. but loss manager right.		my_auto_ptr2->PrintInfo();		//my_auto_ptr->PrintInfo();				// error code. (core dump)	}	// [expirment] auto_ptr: reset(), release();	std::auto_ptr<Base> my_auto_ptr3(new Base(33));	if (my_auto_ptr3.get()) {		//my_auto_ptr3.release();					// error code. load to memory dump.		/*		Base * tmp_ptr = my_auto_ptr3.release();	// right code.		delete tmp_ptr;		*/		my_auto_ptr3.reset();				// reset. enable to free memory	}	// std::pointer_ptr	return 0;}
```
**std::auto_ptr几点说明：**

1. auto_ptr对象赋值，会失去内存管理所有权，使得```my_auto_ptr```悬空，最后使用时会导致程序崩溃。因此，**使用```std::auto_ptr```时，尽量不要使用```operator=```操作符**，如果使用了，不要再使用先前对象。
2. 使用```std::auto_ptr release()```函数时，会发现创建出来的对象没有被析构，导致内存泄漏。这是因为**```release```函数不会释放对象空间，仅仅归还所有权**。释放对象可使用```reset()```函数。
3. ```std::auto_ptr```不能当作参数传递，同时其管理的对象也不能放入```std::vector```等容器中，因为```operator=```问题。

+ 参考链接：http://blog.csdn.net/xt_xiaotian/article/details/5714477


#### [```unique_ptr```](http://www.cplusplus.com/reference/memory/unique_ptr/?kw=unique_ptr)

上面的```std::auto_ptr```在复制构造或者```operator=```时，原先的对象就报废了，因为所有权转移到新对象去了，是程序崩溃的隐患。为了避免此现象发生，```std::unique_ptr```很好的解决了这个问题，其**不提供智能指针的```operator=```功能，必须通过间接或显示的交出所有权（```std::move```, since c++11）**。

```c++
std::unique_ptr<Base> fetch_unique_ptr() {	std::unique_ptr<Base> ptr(new Base(55));	//ptr->info = "construct unique_ptr.";	return ptr;};

int main(int argc, char * agrv[]) {
	/* std::unqiue_ptr */	std::unique_ptr<Base> my_unique_ptr(new Base(44));	my_unique_ptr->info = "hello unique_ptr";	my_unique_ptr->PrintInfo();			// "info.c_str(), 44"	// [expirment] unique_ptr not support 'operator='	/*	std::unique_ptr<Base> my_unique_ptr2 = my_unique_ptr;	// error.	std::unique_ptr<Base> my_unique_ptr3(my_unique_ptr);	// error.	*/	std::unique_ptr<Base> my_unique_ptr3 = std::move(my_unique_ptr);	// ok	my_unique_ptr3->PrintInfo();		// "info.c_str(), 44"		std::unique_ptr<Base> my_unique_ptr4 = fetch_unique_ptr();	// ok. move construct.	my_unique_ptr4->PrintInfo();		// "info: construct unique_ptr, number: 55"	// [expirment] unique_ptr support vector element	std::unique_ptr<Base> my_unique_ptr5(new Base(66));	(*my_unique_ptr5).info = "hello vector!";	std::vector< std::unique_ptr<Base> > vec;	vec.push_back(std::move(my_unique_ptr5));  // std::move operator	vec.at(0)->PrintInfo();	vec[0]->PrintInfo();
	// my_unique_ptr5->PrintInfo();		// error	return 0;}
```

**std::unique_ptr几点说明：**

1. **无法进行复制构造和赋值操作**：意味着无法得到指向同一个对象的两个unique_ptr. 但提供了移动构造赋值和显示赋值功能。
2. **为动态申请的内存提供异常安全**: 可以将动态申请内存的所有权传递给某个函数；从某个函数返回动态申请内存的所有权；
3. **可以作为容器元素**。

```std::auto_ptr```和```std::unique_ptr```都是某一块内存独享所有权的智能指针。实际应用中会出现**某一块内存允许多个智能指针共享**，该当如何？下面的```std::shared_ptr```可以满足该场景。

> 分布式机器学习评估模型指标，即计算auc, logloss等指标时，可以使用```std::shared_ptr```

#### [```shared_ptr```](http://www.cplusplus.com/reference/memory/shared_ptr/?kw=shared_ptr)

+ [reset](http://www.cplusplus.com/reference/memory/shared_ptr/reset/)：重新设置指针。`ptr_.reset(data, del)`


--
## C++11新特性

--
### std::functional

--
### std::bind

--
### [lambda (匿名函数) ](http://blog.csdn.net/augusdi/article/details/11773943)

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
[this]		// 表示值传递方式捕捉当前this指针
```

关于捕捉列表几点说明：

1. **捕捉列表不能重复传递**，比如 ```[=, a], [&, &this]```；
2. **在块作用域之外的lambda函数捕捉列表必须为空**；
3. **在块作用域内的lambda函数仅能捕捉父作用域内的自动变量**，非此作用域或者非自动变量（如静态变量）都会导致编译器出错。

示例说明Lambda表达式用法:

```
std::vector<int> list;
int total = 0;
for (int i = 0; i < 5; ++i) list.push_back(i);
std::for_each(begin(list), end(list), [&total](int x) { 
	total += x;
});			// 计算list中所有元素的总和。
```
变量total被存为lambda函数闭包的一部分。因为total是栈变量（局部变量），这里用total的引用，所以可以改变它的值。

Lambda函数是一个依赖于实现的函数对象类型，这个类型的名字只有编译器知道。**如果用户想把Lambda函数做一个参数来传递，那么形参的类型必须是模版类型或者必须能创建一个```std::function```类似的对象去捕获lambda函数**。使用auto关键字可以帮助存储lambda函数。

```
auto my_lambda_func = [&](int x) { /* ... */ };
auto my_onheap_lambda_func = new auto([=](int x) { /* ... */ });
```

下面例子把匿名函数存储在变量、数组或vector中，并把它们当作命名参数来传递：

```c++
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
### 1. [```undefined reference to `...` ```](http://blog.csdn.net/jfkidear/article/details/8276203)

异常示例1：```undefined reference to `dmlc::Config::Config(std::istream&, bool)' ```

主要原因是```*.cc```程序没有**链接**dmlc库函数，需要在```*.cc```对应```CMakeLists.txt```文件添加 **链接库函数代码，即```target_link_libraries(${exec_name} dmlc)```**，相当于在g++上添加了参数```-ldmlc```

> 类似的问题： ```undefined reference to 'pthread_create'``` 需要添加```-lpthread```

异常示例2: 

```
/home/zhouyong03/workplace/DiMLSys/third_party/root/lib/libdmlc.a(hdfs_filesys.o): In function `dmlc::io::HDFSFileSystem::HDFSFileSystem(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)':hdfs_filesys.cc:(.text+0xb1): undefined reference to `hdfsConnect'/home/zhouyong03/workplace/DiMLSys/third_party/root/lib/libdmlc.a(hdfs_filesys.o): In function `dmlc::io::HDFSFileSystem::GetPathInfo(dmlc::io::URI const&)':hdfs_filesys.cc:(.text+0xcd0): undefined reference to `hdfsGetPathInfo'hdfs_filesys.cc:(.text+0x143a): undefined reference to `hdfsFreeFileInfo
```

编译dmlc-core时，发现是```hdfs_filesys.o```出现了问题，没有链接```hdfsConnect```这些库。主要原因应该是编译时参数配置有问题，要编译出支持hdfs的dmlc-core需要认真研究dmlc-core的编译代码；

异常示例3: ``` undefined reference to `omp_get_num_procs` ```

主要原因是使用了OpenMP，但编译时没有配置OpenMP相关编译环境，需要在CMakeLists.txt中配置```find_package(OpenMP); set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")```条件。

使用OpenMP时，需要在CMake文件中 添加 **编译环境代码**，即：```set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")```.

因此，出现```undefined reference to `...` ```问题时，通常有如下原因：

1. 检查include头文件是否存在，如果没有需要添加```include_directories()```
2. 检查相应的链接库是否存在，如果没有需要```target_link_libraries(${exec_name} dmlc)```;
3. 检查对应的编译环境是否确实，比如pthread, OpenMP都需要在g++编译时，添加对应的编译环境。

--
### 2. [... error while loading shared libraries: *.so : cannot open shared object file: No such file or directory](http://blog.csdn.net/sahusoft/article/details/7388617)

错误提示程序执行时无法加载共享库```*.so```，可能不存在或者没有找到。

解决方案：

1. 首先，用```locate *.so```命令检查共享库是否存在，存过不存在，需要网上下载和安装。如果存在，进入第二步
2. 将```*.so```所对应的目录加入```LD_LIBRARY_PATH```路径中，举例操作：

```
LD_LIBRARY_PATH=${JAVA_HOME}/jre/lib/amd64/servier:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
```

需要配置：```export LD_LIBRARY_PATH=/usr/local/mysql/lib:$LD_LIBRARY_PATH```.

上面的配置在MakeFile中可以直接找到对应的环境变量。在CMakeLists中如何使用呢？ cmake使用环境变量需要使用```ENV```关键词。即: ```$ENV{LD_LIBRARY_PATH}```



