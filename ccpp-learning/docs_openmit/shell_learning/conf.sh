#!/bin/bash
 
#########################################
# File Name: conf.sh
# Author: zhouyong03@meituan.com
# Created Time: 2017-03-08 19:48:15
#########################################

FTP_IP="127.0.01"

NAME="zhouyongsdzh"

declare -a version2ip=(
["${NAME}_H"]="127.0.0.1"
["${NAME}_Z"]="127.0.0.2"
);

TASK="ZHOUYONG"
declare -a task2job=(
["${TASK}_service_version"]="${NAME}:H"
["${TASK}_file_list"]="map,ad,50023,50004"
["${TASK}_batch_size"]=2
);
