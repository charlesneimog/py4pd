cmake_minimum_required(VERSION 3.20)
project(py4pd C)

set(PDCMAKE_DIR
    "${CMAKE_SOURCE_DIR}/Resources/pd.cmake/"
    CACHE PATH "Path to pd.cmake")
include(${PDCMAKE_DIR}/pd.cmake)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(PY4PD_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0") # Adds debugging symbols and disables optimization
    message(STATUS "Building in Debug Mode")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG") # Optimized for Release mode, disables debugging symbols
    message(STATUS "Building in Release Mode")
endif()

# ╭──────────────────────────────────────╮
# │          Object Definitions          │
# ╰──────────────────────────────────────╯
if(NOT DEFINED PYVERSION)
    message(FATAL_ERROR "\nPYVERSION is not defined. Use -DPYVERSION=3.11, for example.")
else()
    message(STATUS "Using Python version: ${PYVERSION}")
endif()

# ───────── Numpy Array Include ───────
if(WIN32)
    execute_process(
        COMMAND powershell -Command "py -${PYVERSION} -c \"import numpy; print(numpy.get_include())\""
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE EXEC_RESULT)
else()
    execute_process(
        COMMAND python${PYVERSION} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE EXEC_RESULT)
endif()

if(EXEC_RESULT)
    message(FATAL_ERROR "Failed get NumPy include directory. Have you used 'pip install numpy'?")
endif()

message(STATUS "NumPy include directory: ${NUMPY_INCLUDE_DIR}")

# Print numpy version
execute_process(
    COMMAND python${PYVERSION} -c "import numpy; print(numpy.__version__)"
    OUTPUT_VARIABLE NUMPY_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE EXEC_RESULT)

message(STATUS "NumPy version: ${NUMPY_VERSION}")

# ──────────── Python Include ────────────
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

# ─────────── Python LD Flags ─────────
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

# split the flags into a list

# ╭──────────────────────────────────────╮
# │           PureData Object            │
# ╰──────────────────────────────────────╯
file(GLOB PY4PD_SOURCES ${CMAKE_SOURCE_DIR}/Sources/*.c)
pd_add_external(py4pd "${PY4PD_SOURCES}")
target_include_directories(py4pd PUBLIC ${NUMPY_INCLUDE_DIR})
target_include_directories(py4pd PUBLIC ${PYTHON_INCLUDE_DIR})
target_include_directories(py4pd PUBLIC "/usr/include/python${PYVERSION}")
target_link_libraries(py4pd PUBLIC "${PYTHON_LD_FLAGS}")

# add debug flag
target_compile_options(py4pd PRIVATE -g)

# ╭──────────────────────────────────────╮
# │              Data Files              │
# ╰──────────────────────────────────────╯
file(GLOB PY4PD_SRC_FILES ${CMAKE_SOURCE_DIR}/Sources/*.c ${CMAKE_SOURCE_DIR}/Sources/*.h)
pd_add_datafile(py4pd "${PY4PD_SRC_FILES}" DESTINATION "Sources")

file(GLOB PY4PD_SRC_FILES ${CMAKE_SOURCE_DIR}/Sources/py4pd/*.py)
pd_add_datafile(py4pd "${PY4PD_SRC_FILES}" DESTINATION "py4pd")

file(GLOB PY4PD_PY_FILES ${CMAKE_SOURCE_DIR}/Sources/py4pd.py)
pd_add_datafile(py4pd "${PY4PD_PY_FILES}")

file(GLOB PY4PD_HELP_FILES ${CMAKE_SOURCE_DIR}/Resources/*.pd)
pd_add_datafile(py4pd "${PY4PD_HELP_FILES}")

pd_add_datafile(py4pd "${CMAKE_SOURCE_DIR}/Resources/py.py")
pd_add_datafile(py4pd "${CMAKE_SOURCE_DIR}/LICENSE")
pd_add_datafile(py4pd "${CMAKE_SOURCE_DIR}/README.md")
pd_add_datafile(py4pd "${CMAKE_SOURCE_DIR}/Resources/Help-Files/" DESTINATION "py4pd/help")
