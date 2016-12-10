## 软件与工具
---

+ author: zhouyongsdzh
+ date: 20161210

### 说明

在mac和linux上的软件与工具，安装过程与使用说明

### 目录

+ 工具
	+ vim
	+ terminator
	+ subtime3与插件
	
+ 软件与程序
	+ 2.1 cmake	
	+ 2.2 OpenMP


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
### 2.1. 软件与程序：CMake 

1. 安装： ```sudo apt-get install cmake```

### 2.2. 软件与程序：OpenMP 
	



