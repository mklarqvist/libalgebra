cmake_minimum_required(VERSION 2.8)
project(libalgebra C CXX)
set(CMAKE_BUILD_TYPE Release)
include_directories(.)

add_executable(benchmark benchmark.cpp)

install(FILES libalgebra.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
