#include "py4pd.h"
#include "pd_module.h"


// ======================================
// ======== py4pd embbeded module =======
// ======================================

PyObject *pdout(PyObject *self, PyObject *args){
    (void)self;
    float f;
    char *string;
    if (PyArg_ParseTuple(args, "f", &f)){
        PyErr_Clear();
    }
    else if (PyArg_ParseTuple(args, "s", &string)){
        char *pd_string = string;
        t_symbol *pd_symbol = gensym(pd_string);
        post("pd_symbol: %s", pd_symbol->s_name);
        PyErr_Clear();
    }
    else if (PyArg_ParseTuple(args, "O", &args)){
        int list_size = PyList_Size(args);
        t_atom *list_array = (t_atom *)malloc(list_size * sizeof(t_atom));
        int i;
        for (i = 0; i < list_size; ++i)
        {
            PyObject *pValue_i = PyList_GetItem(args, i);
            if (PyLong_Check(pValue_i))
            { // DOC: If the function return a list of integers
                long result = PyLong_AsLong(pValue_i);
                float result_float = (float)result;
                list_array[i].a_type = A_FLOAT;
                list_array[i].a_w.w_float = result_float;
            }
            else if (PyFloat_Check(pValue_i)){ // DOC: If the function return a list of floats
                double result = PyFloat_AsDouble(pValue_i);
                float result_float = (float)result;
                list_array[i].a_type = A_FLOAT;
                list_array[i].a_w.w_float = result_float;
            }
            else if (PyUnicode_Check(pValue_i)){ // DOC: If the function return a list of strings
                const char *result = PyUnicode_AsUTF8(pValue_i);
                list_array[i].a_type = A_SYMBOL;
                list_array[i].a_w.w_symbol = gensym(result);
            }
            else if (Py_IsNone(pValue_i)){   
                // DOC: If the function return a list of None
                // post("None");
            }
            else{
                Py_DECREF(pValue_i);
                Py_DECREF(args);
                return NULL;
            }
        }
        PyErr_Clear();
    }
    else{
        PyErr_SetString(PyExc_TypeError, "pdout: argument must be a float or a string"); // Colocar melhor descrição do erro
        return NULL;
    }
    // WARNING: This function is not working yet.
    return PyLong_FromLong(0);
}


// =================================
PyObject *pdprint(PyObject *self, PyObject *args){
    (void)self;
    char *string;
    if (PyArg_ParseTuple(args, "s", &string)){
        post("[py4pd]: %s", string);
        PyErr_Clear();
    }
    else{
        PyErr_SetString(PyExc_TypeError, "print: argument must be a string"); // Colocar melhor descrição do erro
        return NULL;
    }
    return PyLong_FromLong(0);
}

// =================================
PyObject *pderror(PyObject *self, PyObject *args){
    (void)self;
    char *string;
    if (PyArg_ParseTuple(args, "s", &string)){
        post("Not working yet");
    }
    else{
        PyErr_SetString(PyExc_TypeError, "message: argument must be a string"); // Colocar melhor descrição do erro
        return NULL;
    }
    return PyLong_FromLong(0);
    // WARNING: This function is not working yet.
} 


// =================================
PyObject *pdmoduleError;

// =================================
PyMethodDef PdMethods[] = {                                                          // here we define the function spam_system
    {"out", pdout, METH_VARARGS, "Output in out0 from PureData"},                           // one function for now
    {"print", pdprint, METH_VARARGS, "Print informations in PureData Console"},             // one function for now
    {"error", pderror, METH_VARARGS, "Print error in PureData"},                            // one function for now
    {NULL, NULL, 0, NULL}
};

// =================================
struct PyModuleDef pdmodule = {
    PyModuleDef_HEAD_INIT,
    "pd", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
    PdMethods,
    NULL,
    NULL,
    NULL,
    NULL,
};

// =================================

PyMODINIT_FUNC PyInit_pd(void){
    PyObject *m;
    m = PyModule_Create(&pdmodule);
    if (m == NULL)
        return NULL;
    pdmoduleError = PyErr_NewException("spam.error", NULL, NULL);
    Py_XINCREF(pdmoduleError);
    if (PyModule_AddObject(m, "error", pdmoduleError) < 0){
        Py_XDECREF(pdmoduleError);
        Py_CLEAR(pdmoduleError);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}


