## CMake Learning

+ author: zhouyongsdzh@foxmail.com
+ date: 20160516

### 0. 关于cmake

#### 0.1. cmake综述

历史 && 功能 (常规与先进) && 应用场景

#### 0.2. 一个简单的cmake示例

项目目录如下：

```
/home/zhouyong/myhome/2016-Planning/C-CPP/cmake-learning/cmake_1_samples
```

目录下两个文件，分别是：main.cc和**CMakeLists.txt** 内容分别为：

main.cpp文件

```
/* * File Name: main.cpp * Author: zhouyongsdzh@foxmail.com * Created Time: 2016-08-18 22:57:11 */ #include <iostream>using namespace std;int main() {	std::cout << "cmake samples! come on." << std::endl;	return 0;}
```

一个简单的输出语句。**CMakeLists.txt**内容为：

```
CMAKE_MINIMUM_REQUIRED(VERSION 3.2)##### 1. PROJECT NAMEPROJECT(cmake_samples)##### 2. set && initialize infoSET(SRC_LISTS main.cpp) 		# 设置. SET关键词功能为赋值。这里即：SRC_LISTS=main.cppMESSAGE("[INFO] binary dir: ${PROJECT_BINARY_DIR}")MESSAGE("[INFO] source dir: ${PROJECT_SOURCE_DIR}")MESSAGE("[INFO] src_lists: ${SRC_LISTS}")   # 输出内容为 main.cpp##### 3. executableADD_EXECUTABLE(cmake_samples_exec ${SRC_LISTS})
```

然后，在当前目录下建立build构建目录 (外部构建，out-of-source build)，进入，分别执行:

```
mkdir build
cd build
cmake ..			# CMakeLists.txt文件所在目录
make 				# 构建，输出可执行文件
```

这样在build目录下可以看到一个名为```cmake_samples_exec```的可执行文件. 运行可得：

```
zhouyong@ubuntu:~/myhome/2016-Planning/C-CPP/cmake-learning/cmake_1_samples/build$ ./cmake_samples_exec cmake samples! come on.    # 输出结果.
```

至此，一个简单的cmake构建项目就完成了。

#### 0.3. 先睹为快 (cmake语法)

在0.2中通过一个简单的示例，介绍了cmake在项目构建中时如何使用的。CMakeLists.txt文件中一些语法 先简单介绍一下。先说明一点：**cmake关键词不区分大小写，即大小写含义相同**.

+ ```cmake_minimum_required(version 3.2)```

### 