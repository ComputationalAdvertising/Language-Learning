## CPlusPlus Learning Note

+ author: zhouyongsdzh@foxmail.com
+ date: 20160829
issue:

2. static_cast/dynamic_cast使用场景
3. &, * 在参数调用中的使用场景

### 目录
--

#### 1. [基础知识](#1.基础知识)

+ 1.1. [C++关键字](#1.1.C++关键字)

    |[[namespace](#1.1.1.namespace)]|[[typedef](#1.1.2.typedef)]|[[template](#1.1.3.template)]| [[explicit](#1.1.4.explicit)]
    | --- | --- | --- | --- |
    
+ 1.2. [IO系统](#1.2.IO系统) 

#### 2. [数据结构与算法](#2.数据结构与算法)

| [[string](#2.1.string)]| [[vector](#2.2.vector)]| [[iterator](#2.3.iterator)] | [[map](#2.4.map)] | [[struct](#2.5.struct)] | [[math](#2.6.math)] | [[sort](#2.7.sort)] | [[random](#2.8.random)]
| --- | --- | --- | --- | --- | --- | --- | --- |

#### 3. [C++特性](#3.C++特性)

| [[class](#3.1.class)] | [[virtual]()] | [[smart_ptr](#3.3.smart_ptr)] |
| --- | --- | --- |

#### 4. C++11新特性

| [[std::function](#4.1.function)] | [[std::bind]()] | [[lambda]()] | 哦县城
| --- | --- | --- | ---| 

#### 5. [C++项目常用工具](#5.C++项目常用工具)

| [[protobuf](#5.1.protobuf)] | [[glog](#5.2.glog)] | [[gtest](#5.3.gtest)] | [[grpc](#5.4.grpc)] | 
| --- | --- | --- | --- |
|[[c++与java](#5.5.c++与java)] | [[c++与python](#5.5.c++与python)] |
|[[valgrind](#5.3.1.valgrind)] |

#### 6. [C++问题总结](#6.C++问题总结)

+ 6.1. [undefined reference to `...`](#6.1.undefined reference to `...`)

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

```c++
std::vector<int> vec(10);
int * p = vec.data();
```
> vector经验总结<br>
> 1. 赋值效率：<br>
```c++
// 情况1
std::vector<float> vec;  for (int i = 0; i < 100; ++i) { vec.push_back(i); }
// 情况2
std::vector<float> vec(100);  for (int i = 0; i < 100; ++i) { vec[i] = i; }
```
> `情况2`要比`情况1`效率高60%～100%。**在已知vec长度情况下，尽可能使用＝赋值操作**


<h3 id="#2.4.map">2.4. map</h3>

**`#include <unordered_map>`**

c++11里面支持了unordered_map (c++98之前只能用`#include <tr1/unordered_map>`)

+ `operator []: core dump`

```c++
#include <iostream>#include <stdlib.h>#include <string>#include <unordered_map>using namespace std; struct Unit {  std::string name;  uint32_t age;  Unit(std::string name, uint32_t age) : name(name), age(age) {}};int main(int argc, char ** argv) {  // unordered_map & struct  std::unordered_map<std::string, Unit *> ms;  ms.insert(std::make_pair("xx", new Unit("xx", 10)));  ms.insert({{"yy", new Unit("yy", 20)}, {"zz", new Unit("zz", 30)}});  std::cout << "xx: " << ms["xx"]->name << ", " << ms["xx"]->age << std::endl;  if (ms["xy"] == nullptr) {    std::cout << "nullptr" << std::endl;    // 输出  } else {    std::cout << "not nullptr" << std::endl;  }  std::cout << "xy: " << ms["xy"]->name << ", " << ms["xy"]->age << std::endl;       // core dump.    return 0;}
```
如果key在map中不存在，那么会返回默认值（基本类型，如int，string等）或者空指针（nullptr，针对value为对象类型时），**不能直接对map返回值进行进一步操作，否则会报core dump**

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

<h4 id="3.1.class">3.1 class </h4>

+ C++创建对象，new与不new的区别
    
(1). 不new创建对象：对象存储在栈内存中，作用域结束后就会被释放。`Base base = Base();`

优点：不用担心内存泄漏，系统会自动完成内存的释放。缺点：函数中不能返回该对象的指针，因为函数结束后，该对象的内存就被释放了。

(2). 用new创建对象：是存储在堆内存中，作用域结束后不会被释放。除非进程结束或显示调用delete释放。`Base * base = new Base();`

优点：函数中可以放回对象的指针，因为对象在函数结束后不会被释放。缺点：如果管理不当，不delete的话，容易造成内存泄漏。

因此使用[3.3. smart_ptr](#3.3.smart_ptr)去初始化一个对象。

<h4 id="3.3.smart_ptr">3.3. smart_ptr</h4> 

参考[智能指针](http://blog.csdn.net/xt_xiaotian/article/details/5714477)

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

#### `std:shared_ptr<>`

下面代码可以说清楚`unique_ptr`与`shared_ptr`的区别

```c++
int main(int argc, char * argv[]) {  //std::unique_ptr<A> a(A::Create("b", 10));  //std::shared_ptr<A> a(new B("b", 10));  std::unique_ptr<A> a(new B("b", 10));  //A * a = A::Create("b", 10);  //std::unique_ptr<A> a = fetch_unique_ptr("b", 10);  a->Info();  //A * a1 = A::Create("c", 20);  //a1->Info();  return 0;}```

```std::auto_ptr```和```std::unique_ptr```都是某一块内存独享所有权的智能指针。实际应用中会出现**某一块内存允许多个智能指针共享**，该当如何？下面的```std::shared_ptr```可以满足该场景。

> 分布式机器学习评估模型指标，即计算auc, logloss等指标时，可以使用```std::shared_ptr```

#### [```shared_ptr```](http://www.cplusplus.com/reference/memory/shared_ptr/?kw=shared_ptr)

+ [reset](http://www.cplusplus.com/reference/memory/shared_ptr/reset/)：重新设置指针。`ptr_.reset(data, del)`


--
## C++11新特性

--

<h4 id="4.1.function">4.1 std::function</h4>

类模版`std::function`作为一个通用、多态的函数封装。先看它是如何应用的？

```c++
#include <functional>#include <iostream>using namespace std; std::function<float(float label, float pred)> Loss;float CalcLossValue(float label, float pred) {  return label - pred;}auto lambda = [](float label, float pred) -> float { return label-pred; };class Functor {  public:    float operator()(float label, float pred) {      return label - pred;    }};class TestClass {  public:    float ClassMember(float label, float pred) {      return label - pred;    }    static float StaticMember(float label, float pred) {      return label - pred;    }};int main() {  Loss = CalcLossValue;  std::cout << "CalcLossValue  Loss(2.0, 0.9): " << Loss(2.0, 0.9) << std::endl;  Loss = lambda;  std::cout << "lambda Loss(2.0, 0.9): " << Loss(2.0, 0.9) << std::endl;  Functor testFunctor;  Loss = testFunctor;  std::cout << "Functor Loss(2.0, 0.9): " << Loss(2.0, 0.9) << std::endl;  TestClass testObj;  Loss = std::bind(&TestClass::ClassMember, testObj, std::placeholders::_1, std::placeholders::_2);  std::cout << "ClassMember Loss(2.0, 0.9): " << Loss(2.0, 0.9) << std::endl;  Loss = TestClass::StaticMember;  std::cout << "StaticMember Loss(2.0, 0.9): " << Loss(2.0, 0.9) << std::endl;  return 0;}
```

如何理解`std::function`？

std::function对C++中各种可调用实体（普通函数、Lambda表达式、函数指针、以及其它函数对象等）的封装，形成一个新的可调用的std::function对象。

对于各个可调用实体转换成std::function类型的对象，上面的代码都有，运行一下代码，阅读一下上面那段简单的代码。总结了简单的用法以后，来看看一些需要注意的事项：

+ 关于可调用实体转换为std::function对象需要遵守以下两条原则：
    + 转换后的std::function对象的参数能转换为可调用实体的参数；
    + 可调用实体的返回值能转换为std::function对象的返回值。
+ std::function对象最大的用处就是在实现函数回调，使用者需要注意，它不能被用来检查相等或者不相等，但是可以与NULL或者nullptr进行比较。


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

<h2 id="5.C++项目常用工具">5. C++项目常用工具</h2> 

<h3 id="5.1.protobuf">5.1. protobuf</h3>

+ 下载与安装：

```bash
git clone https://github.com/google/protobuf.git
cd protobuf 
./autogen.sh        // 自动安装依赖(gmock等）和生成configure
./configure --prefix=${INSTALLED_DIR}/protobuf/${version}  // 指定安装目录
make                // 漫长的make过程
make check          // check过程也比较慢
make install        // 在安装目录下生成bin, include, lib目录
```

> 由于protobuf是用automake编译的，需要提前安装automake/libtool等工具，使用如下命令：
> 
> `sudo apt-get install autoconf automake libtool curl make g++ unzip`
> 
> 具体可参考：https://github.com/google/protobuf/blob/master/src/README.md

+ protobuf demo


需要注意的地方：

> + **protoc 2.0与3.0版本，函数接口变化比较大，使用pb时需要注意版本问题**。比如：
> 
> ```
> ByteSize();       // for protoc 2.* version
> ByteSizeLong();   // for protoc 3.* version
> ```
> 
> + **protobuf的message在执行clear操作时，是不会对其用到的空间进行回收的，只会对数据进行清理**。
> 
> protobuf message的clear()操作是存在cache机制的，它并不会释放申请的空间，这导致占用的空间越来越大。如果程序中protobuf message占用的空间变化很大，那么最好每次或定期进行清理。这样可以避免内存不断的上涨。

<h3 id="5.3.gtest">5.3. gtest</h3>

#### 1. 编译与使用
源码地址：https://github.com/google/googletest.git

```c++
git clone https://github.com/google/googletest.git
cd gooletest/
cmake .
make 
```
在应用时，只需要用到静态库和头文件。

```
静态库地址：googlemock/gtest/libgtest* .  //  libgtest.a与libgtest_main.a
头文件地址：googletest/include
```
+ 使用

```bash
cd googletest/samples       # 使用示例
cp -r ../../googlemock/gtest/libgtest* .   # copy静态库
g++ -I ../include/ -c sample2_unittest.cc    # 生成sample2_unittest.o 
g++ -I ../include/ sample2.o sample2_unittest.o libgtest_main.a libgtest.a -lpthread -o test2   # 生成可执行文件
./test2    # 运行
```
如果要想运行samples中的其它示例，只需要改sample2_unittest文件即可。
> 也可以直接在`googletest/make`目录下运行make, 得到sample1_unittest可执行文件。

<h3 id="5.5.c++与java">5.5. c++与java</h3>

#### JNI使用初探

Java与C++语言之间的交互是靠JNI调用C++的动态库文件来实现，交互示例：

+ 第1步：先给出一段Java代码，并编译成class文件

```java
package org.mi;
public class TestNative {
  // 声明本地方法，内部实现是C++  public native void say();  public static void main(String[] args) {    System.loadLibrary("mi");       // libmi.so    System.out.println("TestNative");    TestNative tNative = new TestNative();    tNative.say();  }}
```
```bash
javac src/main/java/org/mi/TestNative.java -d classes   # class文件输出至classes目录
# class文件目录：classes/org/mi/TestNative.class
```

+ 第2步：生成头文件（*.h）。根据.class文件

```bash
javah -classpath classes -d include org.mi.TestNative 
# -classpath：指定class路径；-d 指定头文件输出路径
# 生成的头文件为：include/org_mi_TestNative.h
```

+ 第3步：根据头文件，用C++实现

```include/org_mi_TestNative.h```

```c++
/* DO NOT EDIT THIS FILE - it is machine generated */#include <jni.h>/* Header for class org_mi_TestNative */#ifndef _Included_org_mi_TestNative#define _Included_org_mi_TestNative#ifdef __cplusplusextern "C" {#endif/* * Class:     org_mi_TestNative * Method:    say * Signature: ()V */JNIEXPORT void JNICALL Java_org_mi_TestNative_say  (JNIEnv *, jobject);#ifdef __cplusplus}#endif#endif
```

```include/org_mi_TestNative.cc```

```c++
#include <iostream>#include "org_mi_TestNative.h"JNIEXPORT void JNICALL Java_org_mi_TestNative_say(JNIEnv *, jobject) {  std::cout << "org_mi_TestNative.h callback success!" << std::endl;}
```
+  第4步：生成动态库文件

```bash
g++ -fPIC -shared -I /usr/lib/jvm/java-8-oracle/include -I /usr/lib/jvm/java-8-oracle/include/linux -I include -o libmi.so include/org_mi_TestNative.cc 
# -shared 表示生成动态库（-static：静态库）；-I 指定jni.h和jni_md.h路径
```
这样在目录下就生成了libmi.so文件

> 如果不加-fPIC，会报错误：```/usr/bin/ld: /tmp/ccf5WTXs.o: relocation R_X86_64_32 against `.rodata' can not be used when making a shared object; recompile with -fPIC```

+ 第5步：Java调用动态库so文件

```bash
export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
java -classpath classes org.mi.TestNative
TestNativeorg_mi_TestNative.h callback success!
```

#### C++与Java参数对应关系

在JNI调用so时，参数对应关系如下：
todo

<h3 id="5.3.1.valgrind">5.3.1. valgrind</h3>

Valgrind([官网](http://www.valgrind.org/))是运行在Linux上一套基于仿真技术的**程序调试和分析工具**，是公认的最接近Purify的产品，它包含一个内核——一个软件合成的CPU，和一系列的小工具，每个工具都可以完成一项任务——调试，分析，或测试等。**Valgrind可以检测内存泄漏和内存越界，还可以分析cache的使用等，灵活轻巧而又强大**。

valgrind包含的工具如下：

+ Memcheck : 这是valgrind应用最广泛的工具，一个重量级的**内存检查器，能够发现开发中绝大多数内存错误使用情况，比如：内存泄漏，非法指针，越界等**；这也是本文将重点介绍的部分；
+ Callgrind : 检查程序中**函数调用**过程中的问题，比如程序代码覆盖、分析程序性能；
+ Cachegrind ：检查程序中**缓存**使用出现的问题，比如CPU的cache命中率、丢失率，用于代码优化；
+ Helgrind : 它主要用来检查**多线程程序**中出现的**竞争问题**；
+ Massif : 堆栈分析器，指示程序中使用了多少heap内存信息等；
+ Extension : 可以利用core提供的功能，自己编写特定的内存调试工具；

**安装valgrind**

```bash
wget http://www.valgrind.org/downloads/archive/valgrind-3.11.0.tar.bz2      # 下载 (高版本提示错误，内核版本低)
tar xvf valgrind-3.11.0.tar.bz2       # 解压
./configure --prefix=/home/zhouyongsdzh/software/valgrind-3.11.0_build           // configure
make && make install 
# 打开~/.bashrc添加
VALGRAIND_HOME=/home/zhouyongsdzh/software/valgrind-3.11.0_buildPATH=$VALGRAIND_HOME/bin:$PATH

source ~/.bashrc
valgrind -h         # 可以看到如下信息
usage: valgrind [options] prog-and-args  tool-selection option, with default in [ ]:    --tool=<name>             use the Valgrind tool named <name> [memcheck]
```

**MemCheck**

memcheck是最常用的工具，用来检测程序中出现的内存问题，所有对内存的读写都会被检测到，一切对malloc、free、new、delete的调用都会被捕获。所以，它能检测以下问题：
1、对未初始化内存的使用；
2、读/写释放后的内存块；
3、读/写超出malloc分配的内存块；
4、读/写不适当的栈中内存块；
5、内存泄漏，指向一块内存的指针永远丢失；
6、不正确的malloc/free或new/delete匹配；
7、memcpy()相关函数中的dst和src指针重叠。

示例：

```c++
// file: memory_leak.cc
#include <iostream>#include <string>#include <stdint.h>#include <vector>using namespace std; struct FFMEntry {  const void * key_;  uint16_t tag_;  const float * value_;  FFMEntry(const char * key, uint16_t tag, const float * value) :    key_(key), tag_(tag), value_(value) {}    ~FFMEntry() {}};int main(int argc, char ** argv) {  std::vector<float> value = {1, 2, 3, 4};  std::vector<FFMEntry *> vec;  vec.push_back(new FFMEntry("love", 1, value.data()));  vec.push_back(new FFMEntry("this", 2, value.data()));  vec.push_back(new FFMEntry("world", 3, value.data()));  vec.push_back(new FFMEntry("i", 4, value.data()));  std::cout << "create object && and free memory\n";  std::vector<FFMEntry *>().swap(vec);  return 0;}
```

```bash
g++ -std=c++11 memory_leak.cc -o exec
valgrind --leak-check=full ./exec               # leak全部检查
# 运行结果
==16819== Memcheck, a memory error detector==16819== Copyright (C) 2002-2015, and GNU GPL'd, by Julian Seward et al.==16819== Using Valgrind-3.11.0 and LibVEX; rerun with -h for copyright info==16819== Command: ./a.out==16819== create object && and free memory==16819== ==16819== HEAP SUMMARY:==16819==     in use at exit: 72,800 bytes in 5 blocks==16819==   total heap usage: 10 allocs, 5 frees, 73,896 bytes allocated==16819== ==16819== 24 bytes in 1 blocks are definitely lost in loss record 1 of 5==16819==    at 0x4C2E118: operator new(unsigned long) (vg_replace_malloc.c:333)==16819==    by 0x400BFC: main (in /home/zhouyongsdzh/workspace/openmit/openmit/test/a.out)==16819== ==16819== 24 bytes in 1 blocks are definitely lost in loss record 2 of 5==16819==    at 0x4C2E118: operator new(unsigned long) (vg_replace_malloc.c:333)==16819==    by 0x400C44: main (in /home/zhouyongsdzh/workspace/openmit/openmit/test/a.out)==16819== ==16819== 24 bytes in 1 blocks are definitely lost in loss record 3 of 5==16819==    at 0x4C2E118: operator new(unsigned long) (vg_replace_malloc.c:333)==16819==    by 0x400C8C: main (in /home/zhouyongsdzh/workspace/openmit/openmit/test/a.out)==16819== ==16819== 24 bytes in 1 blocks are definitely lost in loss record 4 of 5==16819==    at 0x4C2E118: operator new(unsigned long) (vg_replace_malloc.c:333)==16819==    by 0x400CD4: main (in /home/zhouyongsdzh/workspace/openmit/openmit/test/a.out)==16819== ==16819== 72,704 bytes in 1 blocks are still reachable in loss record 5 of 5==16819==    at 0x4C2DC10: malloc (vg_replace_malloc.c:299)==16819==    by 0x4EC3EFF: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)==16819==    by 0x40106B9: call_init.part.0 (dl-init.c:72)==16819==    by 0x40107CA: call_init (dl-init.c:30)==16819==    by 0x40107CA: _dl_init (dl-init.c:120)==16819==    by 0x4000C69: ??? (in /lib/x86_64-linux-gnu/ld-2.23.so)==16819== ==16819== LEAK SUMMARY:==16819==    definitely lost: 96 bytes in 4 blocks==16819==    indirectly lost: 0 bytes in 0 blocks==16819==      possibly lost: 0 bytes in 0 blocks==16819==    still reachable: 72,704 bytes in 1 blocks==16819==         suppressed: 0 bytes in 0 blocks==16819== ==16819== For counts of detected and suppressed errors, rerun with: -v==16819== ERROR SUMMARY: 4 errors from 4 contexts (suppressed: 2 from 1)
```
> 分析日志：<br>
> 1. 第13行：heap信息提示程序有10次内存分配，但是释放了5次，已经分配的内存量;
> 2. 第40行：definitely lost: 96 bytes in 4 blocks 提示有4块内存泄漏；
> 3. 第31行：`... are still reachable in ...`是参数`--show-leak-kinds=all起到的作用`
> 4. 第47行：ERROR SUMMARY: 提示有4个错误来自4个block。
> 上面内存泄漏的原因是：**vector中的元素是指针类型，使用swap虽然可以释放vector，但是其元素指向的指针没有被释放，即这里vector的4个元素只有new没有delete**.

需要释放掉vector元素的指针：

```c++
#include <iostream>#include <string>#include <stdint.h>#include <vector>using namespace std; struct FFMEntry {  const void * key_;  uint16_t tag_;  const float * value_;  FFMEntry(const char * key, uint16_t tag, const float * value) :    key_(key), tag_(tag), value_(value) {}    ~FFMEntry() {}};int main(int argc, char ** argv) {  std::vector<float> value = {1, 2, 3, 4};  std::vector<FFMEntry *> vec;  vec.push_back(new FFMEntry("love", 1, value.data()));  vec.push_back(new FFMEntry("this", 2, value.data()));  vec.push_back(new FFMEntry("world", 3, value.data()));  vec.push_back(new FFMEntry("i", 4, value.data()));  std::cout << "create object && and free memory\n";    for (int i = 0; i < vec.size(); ++i) {    delete vec[i]; vec[i] = NULL;  }  std::vector<FFMEntry *>().swap(vec);  return 0;}
```
重新检测: 

```bash
g++ -std=c++11 memory_leak.cc -o exec
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes  --log-file=leak.log ./exec
==18274== Memcheck, a memory error detector==18274== Copyright (C) 2002-2015, and GNU GPL'd, by Julian Seward et al.==18274== Using Valgrind-3.11.0 and LibVEX; rerun with -h for copyright info==18274== Command: ./a.out==18274== create object && and free memory==18274== ==18274== HEAP SUMMARY:==18274==     in use at exit: 72,704 bytes in 1 blocks==18274==   total heap usage: 10 allocs, 9 frees, 73,896 bytes allocated==18274== ==18274== 72,704 bytes in 1 blocks are still reachable in loss record 1 of 1==18274==    at 0x4C2DC10: malloc (vg_replace_malloc.c:299)==18274==    by 0x4EC3EFF: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)==18274==    by 0x40106B9: call_init.part.0 (dl-init.c:72)==18274==    by 0x40107CA: call_init (dl-init.c:30)==18274==    by 0x40107CA: _dl_init (dl-init.c:120)==18274==    by 0x4000C69: ??? (in /lib/x86_64-linux-gnu/ld-2.23.so)==18274== ==18274== LEAK SUMMARY:==18274==    definitely lost: 0 bytes in 0 blocks==18274==    indirectly lost: 0 bytes in 0 blocks==18274==      possibly lost: 0 bytes in 0 blocks==18274==    still reachable: 72,704 bytes in 1 blocks==18274==         suppressed: 0 bytes in 0 blocks==18274== ==18274== For counts of detected and suppressed errors, rerun with: -v==18274== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 2 from 1)
```
可以看到第23行`definitely lost: 0 bytes in 0 blocks`和最后一行`ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 2 from 1)` 0错误。

`--log-file=leak.log`表示将检测结果保存至`leak.log`文件中。

> valgrind内存泄漏小结：从提示中可以看到`LEAK SUMMARY`提供了以下几种内存泄漏情况：
> 1. definitely lost: 明确说明已经泄漏了，因为在程序运行完的时候，没有指针指向它, 指向它的指针在程序中丢失了;
> 2. indirectely lost:  间接说明泄漏
> 3. still reachable: 表示泄漏的内存在程序运行完的时候，仍旧有指针指向它，因而，这种内存在程序运行结束之前可以释放。一般情况下valgrind不会报这种泄漏，除非使用了参数 --show-reachable=yes。
> 4. possibly lost: 发现了一个指向某块内存中部的指针，而不是指向内存块头部。这种指针一般是原先指向内存块头部，后来移动到了内存块的中部，还有可能该指针和该内存根本就没有关系，检测工具只是怀疑有内存泄漏
> 5. suppressed: ？

ValGrind其他检测信息说明

+ `Conditional jump or move depends on uninitialised value(s)` : 表明程序中有字符串、结构体成员等变量没有初始化（一般不会有影响）
+ `valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --log-file=leak.log`：将检测结果保存至leak.log中

## C++问题总结

--
<h3 id="6.1.undefined reference to `...`">6.1. undefined reference to `...`</h3>

参考链接：http://blog.csdn.net/jfkidear/article/details/8276203

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

异常示例4：```undefined reference to `spacex::FFM<unsigned long, float>::Predict(dmlc::Row<unsigned long> const&, std::vector<unsigned long, std::allocator<unsigned long> > const&, std::vector<float, std::allocator<float> > const&, std::vector<int, std::allocator<int> > const&)'```

主要原因是这里把模版类分离编译导致。就是把模版类的声明和实现分别放在了头文件和源文件中。而g++本身不支持模版类的分离编译，所有提示找不到方法的具体实现（在*.cc中）。

解决方案：要么不使用模版类，要么把声明和定义放在同一个*.h文件中。参考：http://blog.sina.com.cn/s/blog_6cef0cb50100nb7o.html

因此，出现```undefined reference to `...` ```问题时，通常有如下原因：

1. 检查include头文件是否存在，如果没有需要添加```include_directories()```
2. 检查相应的链接库是否存在，如果没有需要```target_link_libraries(${exec_name} dmlc)```;
3. 检查对应的编译环境是否确实，比如pthread, OpenMP都需要在g++编译时，添加对应的编译环境。
4. 查看对应的类是否是模版类。如果是模版类，不应该有对应的*.cc文件，因为g++不支持模版类的分离编译；

--
### 2. [... error while loading shared libraries: *.so : cannot open shared object file: No such file or directory](http://blog.csdn.net/sahusoft/article/details/7388617)

错误提示程序执行时无法加载共享库```*.so```，可能不存在或者没有找到。

解决方案：

1. 首先，用```locate *.so```命令检查共享库是否存在，如果不存在，需要网上下载和安装。如果存在，进入第二步
2. 将```*.so```所对应的目录加入```LD_LIBRARY_PATH```路径中，举例操作：

```
LD_LIBRARY_PATH=${JAVA_HOME}/jre/lib/amd64/servier:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
```

需要配置：```export LD_LIBRARY_PATH=/usr/local/mysql/lib:$LD_LIBRARY_PATH```.

上面的配置在MakeFile中可以直接找到对应的环境变量。在CMakeLists中如何使用呢？ cmake使用环境变量需要使用```ENV```关键词。即: ```$ENV{LD_LIBRARY_PATH}```

在使用automake编译时，也出现类似的错误：`./openmit: error while loading shared libraries: libprotobuf.so.12: cannot open shared object file: No such file or directory`. automake下的解决方案是？

--
### 3. [...invalid initialization of non-const reference of type...](http://blog.csdn.net/kongying168/article/details/3864756)

```c++
/home/zhouyongsdzh/workspace/openmit/openmit/include/openmit/data.h:24:41: error: invalid initialization of non-const reference of type ‘std::__cxx11::string& {aka std::__cxx11::basic_string<char>&}’ from an rvalue of type ‘std::__cxx11::string {aka std::__cxx11::basic_string<char>}’             std::string & data_format = "auto") {                                         ^
```

错误提示的含义：c++中临时变量不能作为非const的引用参数


--
### 4. [...multiple definition of ...](http://blog.csdn.net/xxxxxx91116/article/details/7446455)

```c++
CMakeFiles/openmit.dir/worker.cc.o: In function `mit::WorkerParam::__MANAGER__()':worker.cc:(.text+0x176): multiple definition of `mit::WorkerParam::__MANAGER__()'   // worker.cc提示多次定义errorCMakeFiles/openmit.dir/cli_main.cc.o:cli_main.cc:(.text+0x3b6): first defined here    // 最早在cli_main.cc中被定义collect2: error: ld returned 1 exit status
```

上面出现错误的原因：把变量的定义（`DMLC_REGISTER_PARAMETER(WorkerParam);`）放在了worker.h文件中，而worker.cc和cli_main.cc都include了worker.h，进行了两次变量的定义，所以提示错误。

解决方案：worker.h中的变量定义放在worker.cc中。如此可避免变量重复定义的问题。

>
1.编译是针对一个一个文件来说的，而链接则是针对一个工程所有的.o文件而言的。
2.#ifndef只是对防止一个文件的重复编译有效
3.全局变量最好在.cpp文件中定义，在.h文件中加上extern申明，因为在.h文件中定义，容易在链接时造成变量重定义

> 如果有“公共函数”需要放在base.h文件中，比如`void NewKey(...) { ... }`，为了防止出现`multiple defination of ...`问题，可以在前面加上`inline`，即`inline void NewKey(...) { ... }`

--
### 5. [ ...error: cannot allocate an object of abstract type ...](http://blog.csdn.net/u012474678/article/details/38866415)

在基类中申明的虚函数，在派生类中必须继承并实现。在new一个派生类时才不会报该错误。

此外，`Unit * base = new SimpleUnit();`而不能是`Unit base = new SimpleUnit();`. 

在C++中，new一个类时，需要用指针接着。参考：[C++创建对象，new与不new的区别](http://blog.csdn.net/autoliuweijie/article/details/50579275)

--
### 6. [... Error in `./xx': free(): invalid pointer: 0x00000000006042e0 ...]()

```c++
*** Error in `./xx': free(): invalid pointer: 0x00000000006042e0 ***======= Backtrace: =========/lib/x86_64-linux-gnu/libc.so.6(+0x777e5)[0x7f3989dea7e5]/lib/x86_64-linux-gnu/libc.so.6(+0x7fe0a)[0x7f3989df2e0a]/lib/x86_64-linux-gnu/libc.so.6(cfree+0x4c)[0x7f3989df698c]./xx[0x4015dc]./xx[0x401d02]./xx[0x401bdb]./xx[0x401397]/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xf0)[0x7f3989d93830]./xx[0x401039]======= Memory map: ========00400000-00403000 r-xp 00000000 08:01 2490760                            /home/zhouyongsdzh/workspace/openmit/openmit/language/xx...7f398a351000-7f398a352000 rw-p 00015000 08:01 398344                     /lib/x86_64-linux-gnu/libgcc_s.so.1...7f398a6d0000-7f398a6d4000 rw-p 00000000 00:00 0 7f398a6d4000-7f398a6fa000 r-xp 00000000 08:01 397534                     /lib/x86_64-linux-gnu/ld-2.23.so...7f398a8fb000-7f398a8fc000 rw-p 00000000 00:00 0 7fffc3ff7000-7fffc4019000 rw-p 00000000 00:00 0                          [stack]7fffc41e1000-7fffc41e3000 r--p 00000000 00:00 0                          [vvar]7fffc41e3000-7fffc41e5000 r-xp 00000000 00:00 0                          [vdso]ffffffffff600000-ffffffffff601000 r-xp 00000000 00:00 0                  [vsyscall]Aborted (core dumped)
```

背景：在工厂方法派生类返回实现, 对应的调用方式：`std::unique_ptr<A> a(A::Create("b", 10));` 报的错误：

```c++
static B * Get(std::string type, int a) {
  //return new B(type, a);
  static B b(type, a);      // stack space  return & b;}
```

如果改成：

```c++
static B * Get(std::string type, int a) {
  return new B(type, a);    // heap space
  //static B b(type, a);        //return & b;}
```

--
### 7. [... error: passing ‘const std::unordered_map<int, mit::Unit*>’ as ‘this’ argument discards qualifiers [-fpermissive] ...]()

具体错误：

```c++
/home/zhouyongsdzh/workspace/openmit/openmit/test/unittests/unittest_openmit_unit.cc: In function ‘void run(const std::unordered_map<int, mit::Unit*>&, int)’:/home/zhouyongsdzh/workspace/openmit/openmit/test/unittests/unittest_openmit_unit.cc:7:37: error: passing ‘const std::unordered_map<int, mit::Unit*>’ as ‘this’ argument discards qualifiers [-fpermissive]   mit::Unit * unit = map_weight_[key];                                     ^In file included from /usr/include/c++/5/unordered_map:48:0,                 from /home/zhouyongsdzh/workspace/openmit/openmit/include/openmit/unit.h:4,                 from /home/zhouyongsdzh/workspace/openmit/openmit/test/unittests/unittest_openmit_unit.cc:1:/usr/include/c++/5/bits/unordered_map.h:667:7: note:   in call to ‘std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc>::mapped_type& std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc>::operator[](const key_type&) [with _Key = int; _Tp = mit::Unit*; _Hash = std::hash<int>; _Pred = std::equal_to<int>; _Alloc = std::allocator<std::pair<const int, mit::Unit*> >; std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc>::mapped_type = mit::Unit*; std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc>::key_type = int]’       operator[](const key_type& __k)
```

问题背景：

```c++
void run(const std::unordered_map<int, mit::Unit * > & map_weight_, int key) {  mit::Unit * unit = map_weight_[key];  unit->SetLinearItem(0.0999);  std::cout << "unit.Linear: " << unit->LinearWeight() << std::endl;}
```

主要原因是：当`const map_weight_`对象调用`operator[]`时，编译器检测出问题。**对一个const对象调用non-const成员函数是不允许的，因为non-const成员函数不保证一定不修改对象。**

编译器在这里做了一个假定，假定`operator[]`试图修改`map_weight_`对象，而与此同时，`map_weight_`是const的，**所有试图修改const对象的都会报error**。

**`unordered_map`的`[]`运算符会在索引项不存在的时候自动创建一个对象，有可能会改变map本身，所以不能在一个const map对象上使用`[]`操作。
**

解决办法：**去掉const**，或者`operator[]`改成const方法（这里比较困难）.

```c++
void run(std::unordered_map<int, mit::Unit * > & map_weight_, int key) { ... }
```

--
### 8. [double free / free: invalid pointer]

`src/learner.cc`出现内存泄漏：

```c++
  /*  mit_float * pvals = map_grad[keys[0]]->Data();  auto offset = map_grad[keys[0]]->Size();  for (size_t i = 1; i < nfeature; ++i) {    std::cout << "Learner::Run i: " << i << std::endl;    memcpy(pvals + offset,         map_grad[keys[i]]->Data(),         map_grad[keys[i]]->Size() * sizeof(mit_float));    offset += map_grad[keys[i]]->Size();  }    std::cout << "nfeature done" << std::endl;  std::vector<mit_float> grad_vals(pvals, pvals + offset);  vals = &grad_vals; */
```

换成下面代码则正常

```c++
  // map_grad_ --> vals  vals->clear();  for (auto i = 0u; i < nfeature; i++) {    mit::Unit * unit = map_grad[keys[i]];    vals->insert(vals->end(),         unit->Data(), unit->Data() + unit->Size());    delete unit;
```

继续跟进问题：

> 注意：<br>
> 1. 一个地址只能由一个指针指向，不能多个指针指向一个地址，否则会出现` double free or corruption (fasttop) ...`问题;

---
### 9. [the following virtual functions are pure within ‘mit::FFM’...]()

具体是`mit::Model`类中有4个纯虚函数(`virtual type method() = 0;`)，而在子类中仅覆写了两个(`override`)，因而提示`下面的虚函数是纯的，必需要覆写~~~`.

> 纯虚函数："virtual type method() ＝0；"； 如果不带`=0`,只有`virtual type method();`则在子类中可以不覆写（不过存在隐患）。

---
### 10. [binding ‘const `value_type` {aka const float}’ to reference of type ‘`mit::mit_float`& {aka float&}’ discards qualifiers]() 

现象：参数为`const std::vector<int> & keys`, 用`keys[i]`参数调用另外一个函数`func(int & key)`报的错误。

解决办法：函数行参加上const即可，即`func(const int & key)`




