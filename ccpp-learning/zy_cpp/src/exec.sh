#!/bin/bash
 
#########################################
# File Name: exec.sh
# Author: zhouyong03@meituan.com
# Created Time: 2016-08-31 17:21:17
#########################################
 
file_name="string_utils.cc"

cmd="g++ ${file_name} -o xx"
${cmd}
cmd="./xx"
${cmd}
