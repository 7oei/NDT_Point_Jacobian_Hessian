# CMakeのバージョンを設定
cmake_minimum_required(VERSION 2.8)
project(AngleDerivatives CXX)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
# add_executable(AngleDerivatives.out AngleDerivatives.cpp)
add_executable(PointDerivatives.out PointDerivatives.cpp)