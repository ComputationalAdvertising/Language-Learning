cmake_minimum_required(VERSION 3.2)

##################################################
# 1. Project
##################################################
project(admm-learner)

##################################################
# 2. Initialize CXXFLAGS && C++11 Activation
##################################################
# 2.1. Initialize CXXFLAGS.
set(CMAKE_CXX_FLAGS                "-Wall -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

# 2.2. Compiler-specific C++11 activation.
MESSAGE(STATUS "[INFO] ${PROJECT_SOURCE_DIR} CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}") 
MESSAGE(STATUS "[INFO] ${PROJECT_SOURCE_DIR} CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

IF ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
	EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
	IF (NOT (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7))
		MESSAGE(FATAL_ERROR "[FATAL_ERROR] ${PROJECT_SOURCE_DIR} ${PROJECT_NAME} requires g++ 4.7 or greater.")
	ELSE ()
		MESSAGE(STATUS "[INFO] ${PROJECT_SOURCE_DIR} ${PROJECT_NAME} g++ compiler version > 4.7 !")
	ENDIF ()
ELSEIF ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
ELSE ()
	MESSAGE(FATAL_ERROR "[FATAL_ERROR] ${PROJECT_SOURCE_DIR} Your C++ compiler does not support C++11.")
ENDIF ()

##################################################
# 3. Dependenced include && library
##################################################
include_directories(${PROJECT_SOURCE_DIR}/include)

link_directories(${PROJECT_SOURCE_DIR}/lib)

set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib")

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
##################################################
# 4. Add Subdirectory
##################################################
add_subdirectory(src)
add_subdirectory(unittests)
##################################################
# 5. Executable Output
##################################################
