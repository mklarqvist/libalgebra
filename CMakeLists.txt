cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(libalgebra C CXX)
set (CMAKE_BUILD_TYPE Release)
set (CMAKE_CXX_FLAGS "-std=c++11")

include_directories(.)

add_executable(benchmark benchmark.cpp)

install(FILES libalgebra.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
