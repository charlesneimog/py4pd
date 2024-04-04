cmake_minimum_required(VERSION 3.15)
include(${CMAKE_CURRENT_SOURCE_DIR}/resources/pd.build/pd.cmake) # pd.build
project(py4pd)

#╭──────────────────────────────────────╮
#│        Python Static Library         │
#╰──────────────────────────────────────╯
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/resources/python-static/)
# set(DOWNLOAD_SOURCES ON)
# set(BUILD_EXTENSIONS_AS_BUILTIN ON)
# set(PY_VERSION 3.11.1)
#
# # define PY_SOURCE_DIR as the python-static source directory
# set(PY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/resources/python-static/)


#╭──────────────────────────────────────╮
#│          Object Definitions          │
#╰──────────────────────────────────────╯
set_pd_external_path(${CMAKE_CURRENT_SOURCE_DIR}/${project_name})
set(SOURCES_DIR ${CMAKE_SOURCE_DIR}/src)
file(GLOB SOURCES ${SOURCES_DIR}/*.cpp)

add_pd_external(py4pd py4pd ${SOURCES})

# ──────────────────────────────────────
if (NOT DEFINED PYVERSION)
    message(FATAL_ERROR "\nPYVERSION is not defined. Use -DPYVERSION=3.11, for example.")
endif()

# ───────── Numpy Array Include ───────
execute_process(
    COMMAND python${PYVERSION} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE EXEC_RESULT
)

if(EXEC_RESULT)
message(FATAL_ERROR "Failed get NumPy include directory. Have you used 'pip install numpy'?")
endif()

set(NUMPY_INCLUDE_DIR ${NUMPY_INCLUDE_DIR} CACHE PATH "NumPy include directory")
message(STATUS "NumPy include directory: ${NUMPY_INCLUDE_DIR}")

# ──────────────────────────────────────
target_include_directories(py4pd PUBLIC ${NUMPY_INCLUDE_DIR})
target_include_directories(py4pd PUBLIC "/usr/include/python${PYVERSION}")
target_link_libraries(py4pd PUBLIC python${PYVERSION})
target_include_directories(py4pd PUBLIC Python)


