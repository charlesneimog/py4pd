cmake_minimum_required(VERSION 3.20)
project(py4pd C)
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_EXTENSIONS OFF) # Use pure C23 without compiler extensions

# ╭──────────────────────────────────────╮
# │               pd.cmake               │
# ╰──────────────────────────────────────╯
cmake_policy(SET CMP0135 NEW)
set(PDCMAKE_FILE ${CMAKE_BINARY_DIR}/pd.cmake)
if(NOT EXISTS ${PDCMAKE_FILE})
    message(STATUS "Downloading pd.cmake")
    file(
        DOWNLOAD https://raw.githubusercontent.com/pure-data/pd.cmake/refs/tags/v0.1.0/pd.cmake ${PDCMAKE_FILE}
        SHOW_PROGRESS
        STATUS DOWNLOAD_STATUS)
endif()
include(${PDCMAKE_FILE})

# ╭──────────────────────────────────────╮
# │          Object Definitions          │
# ╰──────────────────────────────────────╯
if(NOT DEFINED PYVERSION)
    message(FATAL_ERROR "\nPYVERSION is not defined. Use -DPYVERSION=3.11, for example.")
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
file(GLOB PY4PD_SOURCES ${CMAKE_SOURCE_DIR}/Sources/*.c)
pd_add_external(py4pd "${PY4PD_SOURCES}")
target_include_directories(py4pd PUBLIC ${PYTHON_INCLUDE_DIR})
target_include_directories(py4pd PUBLIC "/usr/include/python${PYVERSION}")
target_link_libraries(py4pd PUBLIC "${PYTHON_LD_FLAGS}")
