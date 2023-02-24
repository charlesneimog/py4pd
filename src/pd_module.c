#include "m_pd.h"
#include "py4pd.h"
#include "pd_module.h"

// ======================================
// ======== py4pd embbeded module =======
// ======================================

PyObject *pdout(PyObject *self, PyObject *args){
    
    PyObject *pd_module = PyImport_ImportModule("__main__");
    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, "py4pd");
    t_py *py4pd = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
    outlet_bang(py4pd->out_A);

    // I have this: PyObject_SetAttrString(pd_module, "py4pd", py4pd);
    // Get the py4pd object here

    // py4pd is a pointer to t_py object, get it here

   //
    // if (PyArg_ParseTuple(args, "f", &f)){
    //     PyErr_Clear();
    // }
    // else if (PyArg_ParseTuple(args, "s", &string)){
    //     char *pd_string = string;
    //     t_symbol *pd_symbol = gensym(pd_string);
    //     post("pd_symbol: %s", pd_symbol->s_name);
    //     PyErr_Clear();
    // }
    // else if (PyArg_ParseTuple(args, "O", &args)){
    //     int list_size = PyList_Size(args);
    //     t_atom *list_array = (t_atom *)malloc(list_size * sizeof(t_atom));
    //     int i;
    //     for (i = 0; i < list_size; ++i)
    //     {
    //         PyObject *pValue_i = PyList_GetItem(args, i);
    //         if (PyLong_Check(pValue_i))
    //         { // DOC: If the function return a list of integers
    //             long result = PyLong_AsLong(pValue_i);
    //             float result_float = (float)result;
    //             list_array[i].a_type = A_FLOAT;
    //             list_array[i].a_w.w_float = result_float;
    //         }
    //         else if (PyFloat_Check(pValue_i)){ // DOC: If the function return a list of floats
    //             double result = PyFloat_AsDouble(pValue_i);
    //             float result_float = (float)result;
    //             list_array[i].a_type = A_FLOAT;
    //             list_array[i].a_w.w_float = result_float;
    //         }
    //         else if (PyUnicode_Check(pValue_i)){ // DOC: If the function return a list of strings
    //             const char *result = PyUnicode_AsUTF8(pValue_i);
    //             list_array[i].a_type = A_SYMBOL;
    //             list_array[i].a_w.w_symbol = gensym(result);
    //         }
    //         else if (Py_IsNone(pValue_i)){   
    //             // DOC: If the function return a list of None
    //             // post("None");
    //         }
    //         else{
    //             Py_DECREF(pValue_i);
    //             Py_DECREF(args);
    //             return NULL;
    //         }
    //     }
    //     PyErr_Clear();
    // }
    // else{
    //     PyErr_SetString(PyExc_TypeError, "pdout: argument must be a float or a string"); // Colocar melhor descrição do erro
    //     return NULL;
    // }
    // // WARNING: This function is not working yet.
    return PyLong_FromLong(0);
}


// =================================
PyObject *pdprint(PyObject *self, PyObject *args){
    (void)self;
    char *string;
    if (PyArg_ParseTuple(args, "s", &string)){
        post("[pd.print]: %s", string);
        PyErr_Clear();
    }
    else{
        PyErr_SetString(PyExc_TypeError, "[pd.print] must receive a string"); // Colocar melhor descrição do erro
        return NULL;
    }
    return PyLong_FromLong(0);
}

// =================================
PyObject *pderror(PyObject *self, PyObject *args){
    (void)self;
    char *string;
    if (PyArg_ParseTuple(args, "s", &string)){
        post("[pd.error] Not working yet");
    }
    else{
        PyErr_SetString(PyExc_TypeError, "message: argument must be a string"); // Colocar melhor descrição do erro
        return NULL;
    }
    return PyLong_FromLong(0);
    // WARNING: This function is not working yet.
} 

// =================================
PyObject *pdsend(PyObject *self, PyObject *args){
    (void)self;
    char* receiver;
    char* string;
    float floatNumber;
    int intNumber;
    PyObject *listargs;

    if (PyArg_ParseTuple(args, "ss", &receiver, &string)){
        t_symbol *symbol = gensym(receiver);
        if (symbol->s_thing) {
            pd_symbol(symbol->s_thing, gensym(string));
        }
        else{
            post("[pd.script] object [r %s] not found", receiver);
        }
    }
    else if (PyArg_ParseTuple(args, "sf", &receiver, &floatNumber)){
        t_symbol *symbol = gensym(receiver);
        // convert number to t_float
        if (symbol->s_thing) {
            pd_float(symbol->s_thing, floatNumber);
        }
        else{
            post("[pd.script] object [r %s] not found", receiver);
        }
    }
    else if (PyArg_ParseTuple(args, "si", &receiver, &intNumber)){
        t_symbol *symbol = gensym(receiver);
        if (symbol->s_thing) {
            pd_float(symbol->s_thing, intNumber);
        }
        else{
            post("[pd.script] object [r %s] not found", receiver);
        }
    }
    else if (PyArg_ParseTuple(args, "sO", &receiver, &listargs)){
        if (PyDict_Check(listargs)){
            char error_message[100];
            sprintf(error_message, "[pd.send] received a type 'dict', it must be a list, string, int, or float."); 
            PyErr_SetString(PyExc_TypeError, error_message); // TODO: Check english
            return NULL;
        }
        t_atom *list_array;
        int list_size = PyList_Size(listargs);
        list_array = (t_atom *)malloc(list_size * sizeof(t_atom));
        // check type of args
        int i;
        for (i = 0; i < list_size; ++i)
        {
            PyObject *pValue_i = PyList_GetItem(listargs, i);
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
                char error_message[100];
                sprintf(error_message, "[pd.send] received a type '%s' in index %d of the list, it must be a string, int, or float.", pValue_i->ob_type->tp_name, i);
                PyErr_SetString(PyExc_TypeError, error_message); // TODO: Check english
                Py_DECREF(pValue_i);
                Py_DECREF(args);
                free(list_array);
                return NULL;
            }
        }
        if (gensym(receiver)->s_thing){
            pd_list(gensym(receiver)->s_thing, &s_list, list_size, list_array);
        }
        else{
            post("[pd.script] object [r %s] not found", receiver);
        }
    }
    else{
        char error_message[100];
        // get type of second argument
        PyObject *pValue_i = PyTuple_GetItem(args, 1);
        sprintf(error_message, "[pd.send] received a type '%s', it must be a string, int, or float.", pValue_i->ob_type->tp_name);
        PyErr_SetString(PyExc_TypeError, error_message); // TODO: Check english
        return NULL;
    }

    PyErr_Clear();
    return PyLong_FromLong(0);
}

// =================================
PyObject *pdmoduleError;

// =================================
PyMethodDef PdMethods[] = {                                                        
    {"out", pdout, METH_VARARGS, "Output in out0 from PureData"},   
    {"send", pdsend, METH_VARARGS, "Send message to PureData, it can be received with the object [receive]"},
    {"print", pdprint, METH_VARARGS, "Print informations in PureData Console"},            
    {"error", pderror, METH_VARARGS, "Print error in PureData"},                          
    {NULL, NULL, 0, NULL}
};

// =================================
struct PyModuleDef pdmodule = {
    PyModuleDef_HEAD_INIT,
    "pd", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
    PdMethods, // Methods
    NULL, 
    NULL, 
    NULL, 
    NULL, 
};

// =================================

PyMODINIT_FUNC PyInit_pd(){
    PyObject *m;
    m = PyModule_Create(&pdmodule);
    if (m == NULL){
        return NULL;
    }
    pdmoduleError = PyErr_NewException("spam.error", NULL, NULL);
    Py_XINCREF(pdmoduleError);
    if (PyModule_AddObject(m, "error", pdmoduleError) < 0){
        Py_XDECREF(pdmoduleError);
        Py_CLEAR(pdmoduleError);
        Py_DECREF(m);
        return NULL;
    }
    post("[py4pd] pd module loaded");
    return m;
}


