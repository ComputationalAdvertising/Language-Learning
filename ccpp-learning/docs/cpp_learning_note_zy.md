## CPlusPlus Learning Note

+ author: zhouyongsdzh@foxmail.com
+ date: 20160829

### List

+ 基础知识 (按照字母顺序)
	+ namespace 使用 
+ C++专题
	+ string专题 (```#include<string>```) 
		+ string, char*, char[]转换: http://blog.csdn.net/cogbee/article/details/8931838
	+ io专题
 
+ 项目构建


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
