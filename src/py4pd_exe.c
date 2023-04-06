#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>


#include <Python.h>

#define MAXPDSTRING 1024
char PY4PDHOME[1024];
char PATCHHOME[1024];
char PYMODULE[1024];
char PYFUNCTION[1024];
char PYPACKAGES[1024];

// ============================================
typedef enum{
    PY4PD_FLOAT,
    PY4PD_SYMBOL,
} py4pd_atomtype;

// ============================================
typedef struct _py4pdatom{
    float floatvalue;
    char *symbolvalue;
    py4pd_atomtype a_type;
} py4pd_atom;



// =====================================================================
/**
 * @brief Convert and output Python Values to PureData values
 * @param x is the py4pd object
 * @param pValue is the Python value to convert
 * @return nothing, but output the value to the outlet
 */

py4pd_atom *pyData_to_cData(PyObject *pValue, py4pd_atom *list_array) {
    if (PyList_Check(pValue)) {  // If the function return a list list
        int list_size = PyList_Size(pValue);
        int i;
        int listIndex = 0;
        PyObject *pValue_i = NULL;
        for (i = 0; i < list_size; ++i) {
            pValue_i = PyList_GetItem(pValue, i);
            if (PyLong_Check(pValue_i)) {  // If the function return a list of integers
                float result = (float)PyLong_AsLong(pValue_i); // NOTE: Necessary to change if want double precision
                list_array[listIndex].a_type = PY4PD_FLOAT;
                list_array[listIndex].floatvalue = result;

                listIndex++;
            } 
            else if (PyFloat_Check(pValue_i)) {  // If the function return a list of floats
                float result = PyFloat_AsDouble(pValue_i);
                list_array[listIndex].a_type = PY4PD_FLOAT;
                list_array[listIndex].floatvalue = result;
                listIndex++;
            } 
            else if (PyUnicode_Check(pValue_i)) {  // If the function return a
                const char *result = PyUnicode_AsUTF8(pValue_i); 
                list_array[listIndex].a_type = PY4PD_SYMBOL;
                // save the symbol in the list
                list_array[listIndex].symbolvalue = malloc(strlen(result) + 1);
                strcpy(list_array[listIndex].symbolvalue, result);
                listIndex++;
            } 
            else if (Py_IsNone(pValue_i)) {  // If the function return a list
                                               // of None
            } 
            else {
                printf("[py4pd] py4pd just convert int, float and string! "
                         "Received: %s",
                         Py_TYPE(pValue_i)->tp_name);
                Py_DECREF(pValue_i);
                return 0;
            }
        }
        return list_array;

    } 
    else {
        if (PyLong_Check(pValue)) {
            long result = PyLong_AsLong(pValue);  // If the function return a integer
            list_array[0].a_type = PY4PD_FLOAT;
            list_array[0].floatvalue = result;
            return list_array;
        } 
        else if (PyFloat_Check(pValue)) {
            double result = PyFloat_AsDouble(pValue);  // If the function return a float
            list_array[0].a_type = PY4PD_FLOAT;
            list_array[0].floatvalue = result;
            return list_array;
        } 
        else if (PyUnicode_Check(pValue)) {
            const char *result = PyUnicode_AsUTF8(pValue); // If the function return a string
            list_array[0].a_type = PY4PD_SYMBOL;
            list_array[0].symbolvalue = malloc(strlen(result) + 1);
            strcpy(list_array[0].symbolvalue, result);
            return list_array;
        } 
        else if (Py_IsNone(pValue)) {

        }
        else {
            return 0;
        }
    }
    return 0;
}

// =====================================================================

int main() {
    Py_Initialize();
    int bytes_written;
    const char *pipe_PY4PDHOME = "/tmp/py4pd_PY4PDHOME"; // It sends the Pd home path to the py4pd (exe)
    const char *pipe_PATCHHOME = "/tmp/py4pd_PATCHHOME"; // The name of the named pipe
    const char *pipe_SITEPACKAGES = "/tmp/py4pd_SITEPACKAGES"; // The name of the named pipe
    const char *pipe_PYMODULE = "/tmp/py4pd_PYMODULE"; // The name of the named pipe
    const char *pipe_PYFUNCTION = "/tmp/py4pd_PYFUNCTION"; // The name of the named pipe
    const char *pipe_PYRETURN = "/tmp/py4pd_PYRETURN"; // The name of the named pipe
    const char *pipe_RETURNSIZE = "/tmp/py4pd_RETURNSIZE"; // The name of the named pipe

    // get data from py4pd object
    int fd_PY4PDHOME = open(pipe_PY4PDHOME, O_RDONLY);
    int fd_PATCHHOME = open(pipe_PATCHHOME, O_RDONLY);
    int fd_PYMODULE = open(pipe_PYMODULE, O_RDONLY);
    int fd_PYFUNCTION = open(pipe_PYFUNCTION, O_RDONLY);
    int fd_PYPACKAGES = open(pipe_SITEPACKAGES, O_RDONLY);

    bytes_written = read(fd_PY4PDHOME, PY4PDHOME, sizeof(PY4PDHOME));
    PY4PDHOME[bytes_written] = '\0'; // add null terminator to string
    bytes_written = read(fd_PATCHHOME, PATCHHOME, sizeof(PATCHHOME));
    PATCHHOME[bytes_written] = '\0'; // add null terminator to string
    bytes_written = read(fd_PYMODULE, PY4PDHOME, sizeof(PY4PDHOME));
    PY4PDHOME[bytes_written] = '\0'; // add null terminator to string
    bytes_written = read(fd_PYMODULE, PYMODULE, sizeof(PYMODULE));
    PYMODULE[bytes_written] = '\0'; // add null terminator to string
    bytes_written = read(fd_PYFUNCTION, PYFUNCTION, sizeof(PYFUNCTION));
    PYFUNCTION[bytes_written] = '\0'; // add null terminator to string
    bytes_written = read(fd_PYPACKAGES, PYPACKAGES, sizeof(PYPACKAGES));
    PYPACKAGES[bytes_written] = '\0'; // add null terminator to string

    close(fd_PATCHHOME);
    close(fd_PYMODULE);
    close(fd_PYFUNCTION);
    close(fd_PYPACKAGES);

    char script_file_path[MAXPDSTRING];
    snprintf(script_file_path, MAXPDSTRING, "%s/%s.py", PATCHHOME, PYMODULE);

    char py4pd_libraries[MAXPDSTRING];
    snprintf(py4pd_libraries, MAXPDSTRING, "%s/resources/scripts", PY4PDHOME);

    // check if script file exists
    if (access(script_file_path, F_OK) == -1) {
        printf("py4pd: script file %s does not exist\n", script_file_path);
        int returnPIPE = open(pipe_PYRETURN, O_WRONLY);
        int returnSIZEPIPE = open(pipe_RETURNSIZE, O_WRONLY);
        close(returnPIPE);
        close(returnSIZEPIPE);
        return 1;
    }
    PyObject *sys_path = PySys_GetObject("path");
    PyList_Insert(sys_path, 0, PyUnicode_FromString(PATCHHOME));
    PyList_Insert(sys_path, 0, PyUnicode_FromString(py4pd_libraries));


    // import script file
    PyObject *pName = PyUnicode_DecodeFSDefault(PYMODULE);
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    // open pipe to  send data to pd
    int returnPIPE = open(pipe_PYRETURN, O_WRONLY);
    int returnSIZEPIPE = open(pipe_RETURNSIZE, O_WRONLY);

    if  (pModule != NULL) {
        PyObject *pFunc = PyObject_GetAttrString(pModule, PYFUNCTION);
        if (pFunc && PyCallable_Check(pFunc)) {
            PyObject *pValue = PyObject_CallObject(pFunc, NULL);
            if (pValue != NULL) {
                int pValueSize;
                if (PyList_Check(pValue)) {
                    pValueSize = PyList_Size(pValue);
                }
                else {
                    pValueSize = 1;
                }
                write(returnSIZEPIPE, &pValueSize, sizeof(int));
                py4pd_atom *list_array = malloc(sizeof(py4pd_atom) * pValueSize);
                pyData_to_cData(pValue, list_array);
                write(returnPIPE, list_array, sizeof(py4pd_atom) * pValueSize);
                free(list_array);

            }
            else{
                PyErr_Print();
            }
        }
        else{
            PyErr_Print();
        }
    }
    else{
        PyErr_Print();
    }

    // unlink(pipe_PDHOME);
    unlink(pipe_PYMODULE);
    unlink(pipe_PYFUNCTION);
    
    close(returnPIPE);
    close(returnSIZEPIPE);
    Py_Finalize();
    printf("py4pd: Python interpreter finalized\n");
    return 0;
}

