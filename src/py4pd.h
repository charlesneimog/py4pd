#define PY_SSIZE_T_CLEAN
#include <Python.h>


// Function of spam_system will be called by python like spam.system("ls -l")


// PyArg_ParseTuple() 

static PyObject *spam_system(PyObject *self, PyObject *args){
    const char *command;
    int sts;
    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

static PyMethodDef SpamMethods[] = { // here we define the function spam_system
    {"system", spam_system, METH_VARARGS, "Execute a shell command."}, // one function for now
    {NULL, NULL, 0, NULL}        
};

static struct PyModuleDef spammodule = {
    PyModuleDef_HEAD_INIT,
    "spam",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SpamMethods
};

PyMODINIT_FUNC PyInit_spam(void){
    return PyModule_Create(&spammodule);
}