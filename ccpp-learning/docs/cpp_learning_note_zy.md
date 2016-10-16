## CPlusPlus Learning Note

+ author: zhouyongsdzh@foxmail.com
+ date: 20160829

### 目录

--
#### 基础知识
+ namespace
+ io系统


#### 数据结构与算法
+ string

#### C++特性
+ class与继承
+ virtual
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
### 智能指针

+ ```auto_ptr```
+ ```shared_ptr```


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





