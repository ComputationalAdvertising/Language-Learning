CMAKE_MINIMUM_REQUIRED(VERSION 3.2)

# cmake-a

#PROJECT(cmakea)
#SET(SRC_LISTS src/main.cc)
#MESSAGE("[INFO] source dir: ${PROJECT_SOURCE_DIR}")
#MESSAGE("[INFO] binary dir: ${PROJECT_BINARY_DIR}")
#ADD_EXECUTABLE(cmake ${SRC_LISTS})

# cmake-b
#PROJECT(cmakeb)
#ADD_SUBDIRECTORY(src bin) 
#MESSAGE("[INFO] source dir: ${PROJECT_SOURCE_DIR}")
#MESSAGE("[INFO] binary dir: ${PROJECT_BINARY_DIR}")
#INSTALL(FILES COPYRIGHT README DESTINATION ${PROJECT_SOURCE_DIR})

##################################################
# 1. Project Name
##################################################
PROJECT(cmake-learning)

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
MESSAGE(STATUS "[INFO] CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}") 
MESSAGE(STATUS "[INFO] CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

IF ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
	EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
	IF (NOT (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7))
		MESSAGE(FATAL_ERROR "[FATAL_ERROR] ${PROJECT_NAME} requires g++ 4.7 or greater.")
	ELSE ()
		MESSAGE(STATUS "[INFO] ${PROJECT_NAME} g++ compiler version > 4.7 !")
	ENDIF ()
ELSEIF ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
ELSE ()
	MESSAGE(FATAL_ERROR "[FATAL_ERROR] Your C++ compiler does not support C++11.")
ENDIF ()

##################################################
# 3. Add Subdirectory
##################################################
# cmaked
ADD_SUBDIRECTORY(src bin)
# cmakec
#ADD_SUBDIRECTORY(lib)

##################################################
# other test code
MESSAGE(STATUS "HOME DIR: $ENV{HOME}")
MESSAGE(STATUS "PROJECT_NAME: ${PROJECT_NAME}")

