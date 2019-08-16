cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(libalgebra C CXX)
set (CMAKE_BUILD_TYPE Release)

macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 11)
  endif ()
endmacro(use_cxx11)

# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

include_directories(.)

add_executable(benchmark benchmark.cpp)

install(FILES libalgebra.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
