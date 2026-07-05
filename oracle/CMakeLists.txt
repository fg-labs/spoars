cmake_minimum_required(VERSION 3.12)

project(spoa_oracle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Force spoa's SISD (scalar) alignment engine. spoa's own CMakeLists.txt
# defaults spoa_optimize_for_native to ON, which adds -march=native; on x86-64
# CI hosts that defines __AVX2__ and spoa's AlignmentEngine::Create silently
# picks the SIMD engine instead of SISD. The SIMD backtrack can diverge from
# SISD on score ties, so a "golden" oracle built with -march=native would not
# be a faithful reference for the scalar Rust reimplementation. Disabling this
# (and adding no -march/-msse4.1/-mavx2 flags of our own, and not enabling
# SIMDe) leaves none of __AVX2__ / __SSE4_1__ / SPOA_USE_SIMDE defined, so
# spoa's SIMD engine template compiles to a stub returning nullptr and
# AlignmentEngine::Create falls back to SisdAlignmentEngine::Create.
set(spoa_optimize_for_native OFF CACHE BOOL "" FORCE)
set(spoa_optimize_for_portability OFF CACHE BOOL "" FORCE)
set(spoa_use_simde OFF CACHE BOOL "" FORCE)

# spoa_build_exe / spoa_build_tests already default OFF as a subproject (spoa's
# CMakeLists.txt only turns them on for spoa_main_project), which keeps the
# bioparser/biosoup/googletest/zlib FetchContent fetches from firing. We only
# need the "spoa" library target.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/spoa
                  ${CMAKE_CURRENT_BINARY_DIR}/spoa)

add_executable(spoa_oracle spoa_oracle.cpp)
target_link_libraries(spoa_oracle PRIVATE spoa::spoa)
