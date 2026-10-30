cmake_minimum_required(VERSION 3.18)
project(c2lut LANGUAGES CXX)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT
  RESULT_VARIABLE nanobind_result
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT nanobind_result EQUAL 0)
  message(FATAL_ERROR "nanobind must be installed in the build Python environment")
endif()
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_native NB_STATIC cpp/bindings.cpp)
target_compile_features(_native PRIVATE cxx_std_17)
set_target_properties(_native PROPERTIES CXX_EXTENSIONS OFF)
if(MSVC)
  target_compile_options(_native PRIVATE /W4)
else()
  target_compile_options(_native PRIVATE -Wall -Wextra -Wpedantic)
endif()
install(TARGETS _native LIBRARY DESTINATION c2lut RUNTIME DESTINATION c2lut)
