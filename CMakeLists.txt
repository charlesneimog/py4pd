cmake_minimum_required(VERSION 3.20)
set(CMAKE_C_STANDARD 11)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(py4pd)
endif()

# ╭──────────────────────────────────────╮
# │               pd.cmake               │
# ╰──────────────────────────────────────╯
set(PDCMAKE_FILE ${CMAKE_BINARY_DIR}/pd.cmake)
set(PDCMAKE_VERSION "v0.2.6")
if(NOT EXISTS "${PDCMAKE_FILE}")
    file(DOWNLOAD https://raw.githubusercontent.com/pure-data/pd.cmake/refs/tags/${PDCMAKE_VERSION}/pd.cmake ${PDCMAKE_FILE})
endif()
include(${PDCMAKE_FILE})

# ╭──────────────────────────────────────╮
# │          Object Definitions          │
# ╰──────────────────────────────────────╯
if(NOT DEFINED PYVERSION)
    message(FATAL_ERROR "PYVERSION is not defined. Use -DPYVERSION=3.13, for example.")
else()
    message(STATUS "Using Python version: ${PYVERSION}")
endif()

# ╭──────────────────────────────────────╮
# │            Python Include            │
# ╰──────────────────────────────────────╯
if(WIN32)
    execute_process(
        COMMAND powershell -Command "py -${PYVERSION} -c \"import sysconfig; print(sysconfig.get_paths()['include'])\""
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE EXEC_RESULT)
else()
    execute_process(
        COMMAND python${PYVERSION} -c "import sysconfig; print(sysconfig.get_paths()['include'])"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE EXEC_RESULT)
endif()

if(EXEC_RESULT)
    message(FATAL_ERROR "Failed get Python include directory.")
endif()

# ╭──────────────────────────────────────╮
# │           Python LD Flags            │
# ╰──────────────────────────────────────╯
if(WIN32)
    execute_process(
        COMMAND powershell -Command
                "py -${PYVERSION} -c \"import sysconfig; print(sysconfig.get_config_var('BINDIR'))\""
        OUTPUT_VARIABLE PYTHON_BIN_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE EXEC_RESULT)
    # remove the dot (.) from PYVERSION
    string(REPLACE "." "" PYVERSION_NO_DOT ${PYVERSION})
    set(PYTHON_LD_FLAGS "-L${PYTHON_BIN_DIR} -lpython${PYVERSION_NO_DOT}")
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
    if(APPLE)
        list(GET PYTHON_LD_FLAGS 0 PYTHON_LD_FLAGS)
    endif()
    string(REPLACE " " ";" PYTHON_LD_FLAGS "${PYTHON_LD_FLAGS}")
    list(APPEND PYTHON_LD_FLAGS "-lpython${PYVERSION}")
endif()

# ╭──────────────────────────────────────╮
# │           PureData Object            │
# ╰──────────────────────────────────────╯
file(GLOB PY4PD_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Sources/*.c)
pd_add_external(py4pd "${PY4PD_SOURCES}")

file(GLOB PY4PD_OBJECTS ${CMAKE_CURRENT_SOURCE_DIR}/Sources/py4pd/*.pd_py)
file(GLOB PY4PD_OBJECTS_HELP ${CMAKE_CURRENT_SOURCE_DIR}/Sources/py4pd/*.pd)
file(GLOB PY4PD_EXAMPLES_OBJECTS ${CMAKE_CURRENT_SOURCE_DIR}/Resources/*.pd_py)

pd_add_datafile(py4pd "${PY4PD_OBJECTS}")
pd_add_datafile(py4pd "${PY4PD_OBJECTS_HELP}")
pd_add_datafile(py4pd "${PY4PD_EXAMPLES_OBJECTS}")

target_include_directories(py4pd PUBLIC ${PYTHON_INCLUDE_DIR})
target_include_directories(py4pd PUBLIC "/usr/include/python${PYVERSION}")
target_link_libraries(py4pd PUBLIC "${PYTHON_LD_FLAGS}")
