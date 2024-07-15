cmake_minimum_required(VERSION 3.20)
project(py4pd C)

set(PDCMAKE_DIR
    "${CMAKE_SOURCE_DIR}/Resources/pd.cmake/"
    CACHE PATH "Path to pd.cmake")
include(${PDCMAKE_DIR}/pd.cmake)

# ╭──────────────────────────────────────╮
# │          Object Definitions          │
# ╰──────────────────────────────────────╯
if(NOT DEFINED PYVERSION)
    message(FATAL_ERROR "\nPYVERSION is not defined. Use -DPYVERSION=3.11, for example.")
else()
    message(STATUS "Using Python version: ${PYVERSION}")
endif()


# ───────── Numpy Array Include ───────
execute_process(
        COMMAND python${PYVERSION} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE EXEC_RESULT)

if(EXEC_RESULT)
    message(FATAL_ERROR "Failed get NumPy include directory. Have you used 'pip install numpy'?")
endif()

message(STATUS "NumPy include directory: ${NUMPY_INCLUDE_DIR}")

# ──────────── Python Include ────────────
execute_process(
    COMMAND python${PYVERSION} -c "import sysconfig; print(sysconfig.get_paths()['include'])"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE EXEC_RESULT)


if(EXEC_RESULT)
    message(FATAL_ERROR "Failed get Python include directory.")
endif()

# ─────────── Python LD Flags ─────────
if (WIN32)
    set(PYTHON_LD_FLAGS "-lpython${PYVERSION}")

else()
execute_process(
    COMMAND python${PYVERSION}-config --ldflags
    OUTPUT_VARIABLE PYTHON_LD_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE EXEC_RESULT)

if(EXEC_RESULT)
    message(FATAL_ERROR "Failed get Python include directory.")
endif()

string(REPLACE " " ";" PYTHON_LD_FLAGS ${PYTHON_LD_FLAGS})

endif()
# ──────────────────────────────────────
file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/Sources/*.c)
pd_add_external(py4pd "${SOURCES}")
target_include_directories(py4pd PUBLIC ${NUMPY_INCLUDE_DIR})
target_include_directories(py4pd PUBLIC ${PYTHON_INCLUDE_DIR})
target_link_libraries(py4pd PUBLIC ${PYTHON_LD_FLAGS})
target_link_libraries(py4pd PUBLIC python${PYVERSION})
target_include_directories(py4pd PUBLIC Python)
