CMAKE_MINIMUM_REQUIRED(VERSION 3.2)

##################################################
# 1. ${NAME}_SRC && EXECUTABLE_OUTPUT_PATH
##################################################
SET(ADMM_SRC admm_learner.cpp)
SET(EXEC_PROGRAM_NAME "admm_learner")

##################################################
# 2. Add Include Directories
##################################################
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/include")
MESSAGE(STATUS "[INFO] ${CMAKE_CURRENT_SOURCE_DIR} include dirs: ${PROJECT_SOURCE_DIR}/include")

##################################################
# 3. Add Library Directories
##################################################
LINK_DIRECTORIES("${PROJECT_SOURCE_DIR}/lib")

##################################################
# 4. Add Executable
##################################################
ADD_EXECUTABLE(${EXEC_PROGRAM_NAME} ${ADMM_SRC})

SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")

##################################################
# 5. Target Link Libraries
##################################################
TARGET_LINK_LIBRARIES(${EXEC_PROGRAM_NAME} hello)
#TARGET_LINK_LIBRARIES(main libhello.a libjson_linux-gcc-4.2.1_libmt.a)

