## CMake && MakeFile Learning

+ author: zhouyongsdzh@foxmail.com
+ date: 20160516

---
### 目录

+ [0. cmake与makefile对应关系](#cmake与makefile对应关系)
+ [1. cmake](#1.cmake)
    + [1.1. 关于cmake](#1.1.关于cmake) 
        + [1.1.1. cmake综述](#1.1.1.cmake综述)
        + [1.1.2. cmake简单示例](#1.1.2.cmake综述) 
    + [1.2. cmake语法](#1.2.cmake语法)
        + [1.2.1. cmake指令与关键词](#1.2.1.cmake指令与关键词)
        + [1.2.2. cmake变量](#1.2.2.cmake变量) 
    + [1.3. cmake示例](#1.3.cmake示例)
        + [1.3.1. 如何生成静态库和动态库](#1.3.1.如何生成静态库和动态库) 
        + [1.3.2. 如何生成可执行程序](#1.3.2.如何生成可执行程序)
        + [1.3.3. OpenMIT项目CMakeLists.txt](#1.3.3.OpenMIT项目CMakeLists.txt)

+ [2. makefile](#2.makefile) 

---
<h3 id="cmake与makefile对应关系">0. cmake与makefile对应关系</h3> 

| 功能 | cmake | makefile | 说明 |
| --- | --- | --- | --- |
| C编译器 | `set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC ...)` | `CFLAGS += -fPIC ...` | C编译器选项 |
| C++编译器 | `set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -pthread ...)` | `CXXFLAGS += -fPIC -Wall -pthread...` | C++编译器选项 |
| 添加include | `include_directories(${dir} ...)` | `CXXFLAGS += -I$(dir) ...` | makefile在`CXXFLAGS`里添加 | 
| 添加lib目录 | `link_directories(${dir} ...)` | `LDFLAGS += -L$(dir) ...` | 多个目录用空格分开 |
| 指定依赖库 | `target_link_libraries(${exec_name} ${lib1} ${lib2} ...)` | `LIBS += -l$(lib1) ...` | 多个依赖库用空格分开 | 

---
<h3 id="1.cmake">1. cmake</h3> 

---
<h4 id="1.1.cmake综述">1.1. cmake综述</h4> 


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

<h3 id="1.2.cmake语法">1.2. cmake语法</h3>
---

<h4 id="1.2.1.cmake指令与关键词">1.2.1. cmake指令与关键词</h4>
---

**cmake指令与关键词不区分大小写，大小写含义相同**.

> **注意：这里说的是指令和关键词不分大小写，而不是cmake变量。cmake变量是需要区分大小写的，比如```EXECUTABLE_OUTPUT_PATH``` 就只能大写。谨记！！！**

| 指令 | 含义 |  说明 |
| --- | --- | --- |
| ```cmake_minimum_required(version 3.2)``` | 指定cmake最低版本。这里是3.2版. |
| ```project(project_name)``` | project关键字用于指定项目名称. |
| ```set(var value)``` | 设置。将value赋给变量var. |
| ```message(STATUS/FATAL_ERROR ${str_info})``` | 打印信息 |
| `include_directories()` | 加载include目录 | 使用外部头文件 |
| `link_directories()` | 加载依赖库目录 | 使用共享库 |
| ```add_executable(${exec_name} ${source})``` | 添加可执行文件名称，以及所需要的源文件。|
| `add_subdirectory()` | 添加编译时需要包含的子目录 |  


<h4 id="1.2.2.cmake变量">1.2.2. cmake变量</h4>
---

cmake使用`${}`引用变量，这与Makefile使用`$()`引用变量不同。**在IF等语句中，直接使用变量名而不通过`${}`取值**。cmake自定义变量主要有隐式定义和显士定义两种。

+ PROJECT指令就属于隐式定义，它会间接的定义```<project_name>_BINARY_DIR```和```<project_name>_SOURCE_DIR```两个变量。
+ 显式定义 使用SET指令，可以构建一个自定义变量。


| 变量名 | 含义 |
| --- | --- |
| ```CMAKE_BINARY_DIR```<br> ```PROJECT_BINARY_DIR``` | 如果是in-source编译，指的是工程顶层目录；如果是out-of-source编译，指的是工程编译发生的目录 |
| ```CMAKE_SOURCE_DIR```<br> ```PROJECT_SOURCE_DIR``` | 不管是in-source编译还是out-of-source编译，都是指工程顶层目录 |
| ```CMAKE_CURRENT_SOURCE_DIR``` | 指的是当前处理CMakeLists.txt所在的路径. 如果是src下面的CMakeLists.txt, 则该值为```${PROJECT_SOURCE_DIR}/src``` |
| ```CMAKE_CURRENT_BINARY_DIR``` | 如果是in-source编译，它与```CMAKE_CURRENT_SOURCE_DIR```一致，如果是out-of-source编译，它指的是target编译目录，该值为```${PROJECT_SOURCE_DIR}/build/bin``` |
| ```CMAKE_CURRENT_LIST_FILE``` | 当前CMakeList.txt文件对应的完整路径 |
| ```CMAKE_CURRENT_LIST_LINE``` | 输出CMakeList.txt中该变量所在的行 |
| ```CMAKE_MODULE_PATH``` | 这个变量用来定义自己的cmake模块所在的路径。工程较复杂时，可能需要编写cmake模块，需要用SET指令，将自己的cmake模块路径设置一下。比如```SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)```, 然后通过include指令 调用自己的模块了 |
| `EXECUTABLE_OUTPUT_PATH` | 指定目标二进制的位置（生成的exec）。可以使用set指令重置，如```set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)``` |
| ```LIBRARY_OUTPUT_PATH``` | 指定目标二进制的位置（最终的共享库）。可以使用set指令重置，如```SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)``` |
| ```PROJECT_NAME``` | 返回通过project指令定义的项目名称 | 

编译项目时通常会用到一些环境变量，cmake是如何引用环境变量的呢？ **使用```$ENV{NAME}```指令调用系统的环境变量**。例如：```message(STATUS "java home dir: $EMV{JAVA_HOME}")```。但是该环境变量需要在系统配置中`export JAVA_HOME`一下。

设置环境变量的方式：```SET (ENV{变量名} 值)```

| 环境变量 | 含义 |
| --- | --- |
| `JAVA_HOME` | 项目中涉及到libjvm.so相关依赖，以及java封装cpp时需要使用jni.h时 |
| `LD_LIBRARY_PATH` | 项目共享库路径可以统一放在环境配置`LD_LIBRARY_PATH`中 | 

<h3 id="1.3.cmake示例">1.3. cmake示例</h3>
---

<h4 id="1.3.1.如何生成静态库和动态库">1.3.1. 如何生成静态库和动态库</h4>
---

<h4 id="1.3.2.如何生成可执行程序">1.3.2. 如何生成可执行程序</h4>
---

<h4 id="1.3.3.OpenMIT项目CMakeLists.txt">1.3.3. OpenMIT项目CMakeLists.txt</h4>
---

```
cmake_minimum_required(VERSION 3.2)project(openmit)find_package(OpenMP)set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fPIC")set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")# Make sure compiler-specific support C++11message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")	execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)	if (NOT (GCC_VERSION VERSION_GREATER 4.6 OR GCC_VERSION VERSION_EQUAL 4.6))		message(FATAL_ERROR "${PROJECT_NAME} project requires g++ 4.6 or greater.")    else ()        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")	endif()elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")else ()	message(FATAL_ERROR "Your C++ compiler does not support C++11.")endif ()# Dependenced include && libraryinclude_directories(  ${PROJECT_SOURCE_DIR}/include  ${PROJECT_SOURCE_DIR}/third_party/include  $ENV{JAVA_HOME}/include)link_directories(  ${PROJECT_SOURCE_DIR}/third_party/lib  ${PROJECT_SOURCE_DIR}/lib  $ENV{JAVA_HOME}/jre/lib/amd64/server)set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib")set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib")# Add Subdirectoryadd_subdirectory(src)add_subdirectory(test)add_subdirectory(test/unittests)#add_subdirectory(language)
```

注意：

1. 一定要保证变量名大写。
2. 因为是按照顺序执行，需要提高告知系统最终可执行文件的存储位置：```set(EXECUTABLE_OUTPUT_PATH ~)```置于```add_subdirectory```之前 才有效。

---
<h3 id="2.makefile">2. makefile</h3>
--- 

