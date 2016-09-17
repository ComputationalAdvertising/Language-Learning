#!/bin/bash
 
#########################################
# File Name: build.sh
# Author: zhouyongsdzh@foxmail.com
# Created Time: 2016-08-21 17:12:50
#########################################
 

if [ -d build ]; then
	cd build && make clean
	rm -rf ./*
else
	mkdir build
	cd build
	echo "2"
fi

cmake ..
make 
