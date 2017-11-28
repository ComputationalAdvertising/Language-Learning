## 软件与工具
---

+ author: zhouyongsdzh
+ date: 20161210

### 说明

在mac和linux上的软件与工具，安装过程与使用说明

### 目录

+ 1. 工具
	+ 1.1. vim
	+ 1.2. terminator
	+ 1.3. subtime3与插件 
	+ [1.4. protobuf](#1.4.protobuf)
	+ [1.5. atom](#1.5.atom)
	+ [1.6. maven](#1.6.maven)
	
+ 软件与程序
	+ [2.1. cmake](#2.1.cmake)
	   + http://www.linuxidc.com/Linux/2014-02/97363.htm
	+ 2.2. openmp
	+ [2.3. m4/autoconf/automake/pkg-config](#2.3.automake)


### 工具: vim

1. Ubuntu安装vim74：一条命令
2. .vimrc和.vim目录插件等，按照自己电脑mac系统；

### 工具：terminator1. 安装：``` sudo apt-get install terminator```
2. Linux环境好用的终端工具### 工具：subtime3

1. markdown插件

subtime3支持markdown，需要安装两个插件：`Markdown Editing`(提供高亮等功能)和`Markdown Preview`(生成HTML等)。

mac上安装，`command + shift + p`搜索‘install’后，分别选择`Markdown Editing`和`Markdown Preview`并安装。重启subtime3，建立以`.md`为后缀的文件 即可使用。

**设置语法高亮和mathjax支持**：在`Preferences` -> `Package Settings` -> `Markdown Preview` -> `Setting - User`中添加如下参数:

```
{
    /*
        Enable or not mathjax support.
    */
    "enable_mathjax": true,

    /*
        Enable or not highlight.js support for syntax highlighting.
    */
    "enable_highlight": true,
}
```

subtime3上安装markdown插件，参考链接：http://www.cnblogs.com/IPrograming/p/Sublime-markdown-editor.html

**因为macdown可以正常使用业内跳转，所以subtime3+markdown插件没有使用**。 


<h4 id="1.4.protobuf">1.4. protobuf</h4>
---mac环境安装：1. 下载最新版本：[protobuf-3.1.0.tar.gz](https://github.com/google/protobuf/archive/v3.1.0.tar.gz)  ([protobuf-2.6.1.tar.gz](https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz)不支持map结构，所以最好安装最新版本)

2. 解压与安装 
    
    v3.0.0以上版本使用automake最为构建工具，首先要检查是否已经安装autoconf/automake/libtool三个工具。`which automake`. 如果没有需要安装：`brew install autoconf automake libtool`.
    
    ```
    tar -zxvf ~/software/protobuf-3.1.0.tar.gz
    ./autogen.sh
    ./configure --prefix=/usr/local/third_party/protobuf-3.1.0
    make && make check && make install
    ```
    
3. 环境配置
    
    在 ~/.zshrc后面添加：

    ```
    PROTOBUF_HOME=/usr/local/third_party/protobuf-3.1.0 
    export PATH=${PROTOBUF_HOME}/bin:${PATH}
    export LD_LIBRARY_PATH=${PROTOBUF_HOME}/lib:$LD_LIBRARY_PATH
    ```
    
4. 验证安装：`which protoc` / `protoc --version`

Intellij IDEA 安装 Protobuf 插件

1. 进入`Preferences -> Plugins -> Browse repositories`，搜索`Google Protocal Buffer Support` 安装后重启；
2. 配置路径。PB插件安装完毕之后，需要为插件配置 PB 的编译路径，也就是在上一步中我们安装的 protoc。`Preferences -> Build,Execution,Deployment -> Compiler -> 填充/usr/local/third_party/protobuf-3.1.0/bin/protoc` 如此插件配置完成。
3. 如何在项目中使用 PB 插件来快速生成 Java 文件？
    
    + 进入`File -> Project Structure -> Facets添加Protobuf Facets`;
    + 然后点击`Modules`在`Protobuf Facets`对应的`Output source directory`填写`src/main/java`对应的绝对路径；
    + 根据PB描述文件，生成Java文件：
        
    ```
        syntax = "proto3";

        option java_package = "com.openmit.core.offline.protobuf";
        option java_outer_classname = "CtrProto";

        message CtrInfo {
            int32 imp = 1;       // impression nums
            int32 clk = 2;       // click nums
            double ctr = 3;    // click throght rate
        }
    ```
        
        右键选择`Compiler *.proto`即可在指定的package中看到java文件。

参考地址：http://zhuliangliang.me/2015/01/12/protobuf_in_java/

**⚠️：java使用bp时一定要保证版本一致性，要么全用proto2，要么全用proto3. 我这里因为要用到map结构，所以全部用proto3(包括语法、pb-java版本和protoc).**

<h4 id="1.5.atom">1.5. atom</h4>


+ mac安装:

```
1. 下载：https://atom.io/download/mac
2. 直接点击atom*.dmg 安装即可
```

+ 插件管理：

安装好Atom以后你可以通过在命令行中使用`apm`命令来安装管理插件

```
which apm       // 查看命令
apm help install    // 查看install安装命令具体信息
apm search terminate    // 搜索与terminate相关的插件
apm install terminate-plus  // 安装terminate-plus插件 (已经废弃，不可用)
安装好提示：Installing terminal-plus to /Users/zhouyong03/.atom/packages ✓
卸载：apm uninstall terminal-plus
```

+ 必装插件

atom默认会安装77个插件，包括autocomplete自动补全等，下面的插件仍然非常好用。

命令行安装方法：`apm install <package_name>`

```
1. platformio-ide-terminal      // 终端
2. minimap          // 缩略代码
3. vim-mode         // 支持vim
4. ex-mode              
5. relative-numbers // vim相对行号
```


+ ubuntu安装：

```bash
sudo add-apt-repository ppa:webupd8team/atom    // [ENTER]
sudo apt-get update  
sudo apt-get install atom
```<h4 id="1.6.maven">1.6. maven</h4>

+ ubuntu安装

```bash
cd ~/software
// 官网下载最新版：
wget http://mirrors.tuna.tsinghua.edu.cn/apache/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz
tar zxvf apache-maven-*-bin.tar.gz
vi ~/.bashrc 
// 设置环境变量
# maven homeM2_HOME=/home/zhouyongsdzh/software/apache-maven-3.3.9export PATH=${M2_HOME}/bin:$PATH
source ~/.bashrc
mvn --version       // 检测是否安装成功
// 创建.m2目录，从${MAC_HOME}/~/.m2/settings.xml中复制一份至该目录
mkdir ~/.m2 && cp ${MAC_HOME}/.m2/settings.xml .
```### 2. 软件### 2.1. cmake

1. 安装： ```sudo apt-get install cmake```

### 2.2. openmp 
	

### 2.3. m4/autoconf/automake/pkg-config

安装`automake`相关构建工具，需要执行：

```
# ubuntu下面
sudo apt-get install autoconf  (ubuntu)
sudo apt-get install libtool (ubuntu)
# centos下面
sudo yum install autoconf (centos)
sudo yum install libtool (centos)
```



4. 安装pkg-config

    下载最新版本：https://pkg-config.freedesktop.org/releases/pkg-config-0.29.tar.gz
    
    ```
    tar -zxvf ~/software/pkg-config-0.29.tar.gz
    ./configure --with-internal-glib --prefix=/usr/local/third_party/pkg-config/0.29
    make && make install
    ```
    
    环境配置：(mac) ~/.zshrc 添加``
    
    
    


