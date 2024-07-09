cmake_minimum_required(VERSION 3.20)

project(py4pd)
set(CMAKE_CXX_STANDARD 20)

set(PDCMAKE_DIR
    "${CMAKE_SOURCE_DIR}/Resources/pd.cmake/"
    CACHE PATH "Path to pd.cmake")
include(${PDCMAKE_DIR}/pd.cmake)

# ╭──────────────────────────────────────╮
# │          Object Definitions          │
# ╰──────────────────────────────────────╯
if(NOT DEFINED PYVERSION)
  message(
    FATAL_ERROR "\nPYVERSION is not defined. Use -DPYVERSION=3.11, for example."
  )
endif()

# ───────── Numpy Array Include ───────
execute_process(
  COMMAND python${PYVERSION} -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE EXEC_RESULT)

if(EXEC_RESULT)
  message(
    FATAL_ERROR
      "Failed get NumPy include directory. Have you used 'pip install numpy'?")
endif()

set(NUMPY_INCLUDE_DIR
    ${NUMPY_INCLUDE_DIR}
    CACHE PATH "NumPy include directory")
message(STATUS "NumPy include directory: ${NUMPY_INCLUDE_DIR}")

# ──────────────────────────────────────
file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/Sources/*.cpp)
pd_add_external(py4pd "${SOURCES}")
target_include_directories(py4pd PUBLIC ${NUMPY_INCLUDE_DIR})
target_include_directories(py4pd PUBLIC "/usr/include/python${PYVERSION}")
target_link_libraries(py4pd PUBLIC python${PYVERSION})
target_include_directories(py4pd PUBLIC Python)
