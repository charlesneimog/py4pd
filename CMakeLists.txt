cmake_minimum_required(VERSION 3.25)
project(py4pd)

set(PDCMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Resources/pd.cmake")
include(${PDCMAKE_DIR}/pd.cmake)

# Determine Python include directory dynamically
if(WIN32)
    execute_process(
        COMMAND powershell -Command "py -${PYVERSION} -c \"import sysconfig; print(sysconfig.get_paths()['include'])\""
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE EXEC_RESULT)

    execute_process(
        COMMAND powershell -Command "py -${PYVERSION} -c \"import numpy; print(numpy.get_include())\""
        OUTPUT_VARIABLE NUMPY_INCLUDES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE EXEC_RESULT)
else()
    execute_process(
        COMMAND python${PYVERSION} -c "import sysconfig; print(sysconfig.get_paths()['include'])"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE EXEC_RESULT)

    execute_process(
        COMMAND python${PYVERSION} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE EXEC_RESULT)
endif()

find_package(pybind11 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${NUMPY_INCLUDES})

# Define the executable
file(GLOB_RECURSE PY4PD_SOURCES "Sources/*.cpp")
pd_add_external(py4pd "${PY4PD_SOURCES}")
target_link_libraries(py4pd PRIVATE pybind11::embed)

# Set target properties
set_target_properties(
    py4pd
    PROPERTIES CXX_STANDARD 11
               CXX_STANDARD_REQUIRED YES
               CXX_EXTENSIONS NO)
