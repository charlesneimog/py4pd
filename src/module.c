#include "py4pd.h"
#include "module.h"


// ======================================
// ======== py4pd embbeded module =======
// ======================================

PyObject *pdout(PyObject *self, PyObject *args){
    (void)self;
    float f;
    char *string;
    if (PyArg_ParseTuple(args, "f", &f)){
        outlet_float(py4pd_object->out_A, f);
        PyErr_Clear();
    }
    else if (PyArg_ParseTuple(args, "s", &string)){
        char *pd_string = string;
        t_symbol *pd_symbol = gensym(pd_string);
        outlet_symbol(py4pd_object->out_A, pd_symbol);
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
                pd_error(py4pd_object, "[py4pd] py4pd just convert int, float and string!\n");
                pd_error(py4pd_object, "[py4pd] The value received is of type %s", Py_TYPE(pValue_i)->tp_name);
                Py_DECREF(pValue_i);
                Py_DECREF(args);
                return NULL;
            }
        }
        outlet_list(py4pd_object->out_A, 0, list_size, list_array);
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
        pd_error(py4pd_object, "Ocorreu um erro");
    }
    else{
        PyErr_SetString(PyExc_TypeError, "message: argument must be a string"); // Colocar melhor descrição do erro
        return NULL;
    }
    return PyLong_FromLong(0);
    // WARNING: This function is not working yet.
} 
