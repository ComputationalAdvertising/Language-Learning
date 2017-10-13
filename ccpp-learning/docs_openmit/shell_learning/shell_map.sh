#!/bin/bash
 
source conf.sh

service_version=${task2job["${TASK}_service_version"]}
PUSH_FILE_LIST=${task2job["${TASK}_file_list"]}
PUSH_FILE_ARR=(${PUSH_FILE_LIST})
echo "service_version: ${service_version}"
echo "push_file_list: ${PUSH_FILE_LIST}"
echo "push_file_arr: ${PUSH_FILE_ARR}"

