cmake_minimum_required(VERSION 2.8)
project(ndt_sample)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native -fopenmp ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -fopenmp ")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  add_definitions(-DCOMPILEDWITHC11)
  message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  add_definitions(-DCOMPILEDWITHC0X)
  message(STATUS "Using flag -std=c++0x.")
else()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


if(NOT DEFINED python_version)
  set(python_version 3)
endif()

if(python_version STREQUAL 3)
  find_package(PythonLibs 3 REQUIRED)
  find_package(Boost COMPONENTS python38 numpy38 REQUIRED)
elseif(python_version STREQUAL 2)
  find_package(PythonLibs 2.7 REQUIRED)
  find_package(Boost COMPONENTS python numpy REQUIRED)
else()
  message("-Dpython_version=2 or -Dpython_version=3")
  EXIT()
endif()

message(${Boost_LIBRARIES})
message(${PYTHON_INCLUDE_DIRS})
message(${PYTHON_LIBRARIES})

find_package(Eigen3 REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}
  include
  ${PYTHON_INCLUDE_DIRS}
  )

add_library(${PROJECT_NAME} SHARED
  src/ndt_sample.cpp
  )

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
)
