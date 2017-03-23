#!/bin/bash
 
#########################################
# File Name: shell_map.sh
# Author: zhouyong03@meituan.com
# Created Time: 2017-03-08 19:47:51
#########################################
 
source conf.sh

service_version=${task2job["${TASK}_service_version"]}
PUSH_FILE_LIST=${task2job["${TASK}_file_list"]}
PUSH_FILE_ARR=(${PUSH_FILE_LIST})
echo "service_version: ${service_version}"
echo "push_file_list: ${PUSH_FILE_LIST}"
echo "push_file_arr: ${PUSH_FILE_ARR}"

