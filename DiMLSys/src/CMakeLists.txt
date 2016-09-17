cmake_minimum_required(VERSION 3.2)

##################################################
# 1. ${NAME}_SRC
##################################################
aux_source_directory(io IO_SRCS)
aux_source_directory(updater UPDATER_SRCS)

set(ADMM_SRCS 
	${IO_SRCS}
	${UPDATER_SRCS})

##################################################
# 2. Shared && Static Libraries
##################################################
## 2.1. shared libraries
set(shared_lib_name admm)
add_library(${shared_lib_name} SHARED ${ADMM_SRCS})
#set_target_properties(${shared_lib_name} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
#set_target_properties(${shared_lib_name} PROPERTIES VERSION 1.1 SOVERSION 1)

## 2.2. static libraries
set(static_lib_name admm_static)
add_library(${static_lib_name} STATIC ${ADMM_SRCS})
set_target_properties(${static_lib_name} PROPERTIES OUTPUT_NAME "${shared_lib_name}")
set_target_properties(${static_lib_name} PROPERTIES CLEAN_DIRECT_OUTPUT 1)

## 2.3. get target properties
get_target_property(output_value ${static_lib_name} OUTPUT_NAME)
message(STATUS "This is the ${static_lib_name}, output_name: ${output_value}")

##################################################
# 3. Executable 
##################################################
set(exec_name "admm_learner")

add_executable(${exec_name} "admm_learner.cpp")

target_link_libraries(${exec_name} ${shared_lib_name})

