## TensorFlow

+ author: zhouyongsdzh
+ date: 2016 Q2

### 目录 

+ [1. 先睹为快](#1.先睹为快)
    + [1.1. 安装](#1.1.安装) 

<h2 id="1.先睹为快">1. 先睹为快</h2>

<h3 id="1.1.安装">1.1.安装与使用</h3>

以Ubuntu CPU环境为例，参考官网：https://www.tensorflow.org/install/install_linux

TensorFlow在Linux环境下提供了4种安装方式，分别是：

+ virtualenv
+ native pip
+ Dockor

安装：这里以virtualenv方式为例：

```bash
sudo apt-get install python-pip python-dev python-virtualenv
virtualenv --system-site-packages targetDirectory 
source ${targetDirectory}/bin/activate
pip install --upgrade tensorflow
```

> 1. 以上步骤安装失败，重复尝试几次即可；
> 2. `targetDirectory`目录：`~/software/tensorflow`. 

使用方式：

+ 每次要使用tensorflow时，都需要激活虚拟环境。即`source ${targetDirectory}/bin/activate`；
+ 使用完毕后，使用`deactivate`命令 退出虚拟环境；

Uninstalling：`rm -rf ${targetDirectory}`

### 1.2 入门

参考：https://www.tensorflow.org/get_started/get_started
