#include "m_pd.h"
#include "py4pd.h"
#include "py4pd_utils.h"
#include "py4pd_pic.h"  
#include "pylibraries.h"

// ======================================
// ======== py4pd embbeded module =======
// ======================================
PyObject *pdout(PyObject *self, PyObject *args, PyObject *keywords){
    (void)self;
    float f;
    char *string;
    t_symbol *symbol;
    symbol = gensym("list");
    int keyword_arg = 0;




    // check if there is a keyword argument
    if (keywords == NULL) {
        PyErr_Clear();
    } else {
        PyObject *anything_symbol = PyDict_GetItemString(keywords, "symbol");
        if (anything_symbol == NULL) {
            PyErr_Clear();
        } 
        else {
            if (PyUnicode_Check(anything_symbol)) {
                const char *result = PyUnicode_AsUTF8(anything_symbol);
                symbol = gensym(result);
                keyword_arg = 1;
            } 
            else {
                PyErr_SetString(PyExc_TypeError, "[Python] pd.out keyword argument 'symbol' must be a string.");  // Colocar melhor descrição do erro
                return NULL;
            }
        }
    }

    // ================================
    PyObject *pd_module = PyImport_ImportModule("__main__");
    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, "py4pd");
    t_py *py4pd = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
    // ================================

    if (PyArg_ParseTuple(args, "f", &f)) {
        if (keyword_arg == 1) {
            t_atom *list_array = (t_atom *)malloc(1 * sizeof(t_atom));
            list_array[0].a_type = A_FLOAT;
            list_array[0].a_w.w_float = f;
            outlet_anything(py4pd->out_A, symbol, 1, list_array);
            free(list_array);
        } 
        else {
            outlet_float(py4pd->out_A, f);
        }
    } 
    else if (PyArg_ParseTuple(args, "s", &string)) {    
        if (keyword_arg == 1) {
            t_atom *list_array = (t_atom *)malloc(1 * sizeof(t_atom));
            list_array[0].a_type = A_SYMBOL;
            list_array[0].a_w.w_symbol = gensym(string);
            outlet_anything(py4pd->out_A, symbol, 1, list_array);
            free(list_array);
        } 
        else {
            outlet_anything(py4pd->out_A, gensym(string), 0, NULL);
        }


        outlet_symbol(py4pd->out_A, gensym(string));  // TODO: make this output without the symbol, just like fromsymbol.
    } 
    else if (PyArg_ParseTuple(args, "O", &args)) {
        int list_size = PyList_Size(args);
        t_atom *list_array = (t_atom *)malloc(list_size * sizeof(t_atom));
        int i;
        for (i = 0; i < list_size; ++i) {
            PyObject *pValue_i = PyList_GetItem(args, i);
            if (PyLong_Check(pValue_i)) {  // If the function return a list of integers
                long result = PyLong_AsLong(pValue_i);
                float result_float = (float)result;
                list_array[i].a_type = A_FLOAT;
                list_array[i].a_w.w_float = result_float;
            } 
            else if (PyFloat_Check(pValue_i)) {  // If the function return a list of floats
                double result = PyFloat_AsDouble(pValue_i);
                float result_float = (float)result;
                list_array[i].a_type = A_FLOAT;
                list_array[i].a_w.w_float = result_float;
            } 
            else if (PyUnicode_Check(pValue_i)) {  // If the function return a list of strings
                const char *result = PyUnicode_AsUTF8(pValue_i);
                list_array[i].a_type = A_SYMBOL;
                list_array[i].a_w.w_symbol = gensym(result);
            } 
            else if (Py_IsNone(pValue_i)) {
                // If the function return a list of None
            } 
            else {
                Py_DECREF(pValue_i);
                Py_DECREF(args);
                return NULL;
            }
        }
        outlet_anything(py4pd->out_A, symbol, list_size, list_array);
        free(list_array);
        PyErr_Clear();
    } else {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.out argument must be a list, float or a string");  // Colocar melhor descrição do erro
        return NULL;
    }
    return PyLong_FromLong(0);
}

// =================================
PyObject *pdprint(PyObject *self, PyObject *args, PyObject *keywords) {
    (void)self;
    char *string;
    int printPreffix = 1;

    if (keywords == NULL) {
        printPreffix = 1;
        PyErr_Clear();
    } 
    else {
        printPreffix = PyDict_Contains(keywords, PyUnicode_FromString("show_prefix"));
        if (printPreffix == -1) {
            post("error");
        } 
        else if (printPreffix == 1) {
            PyObject *resize_value = PyDict_GetItemString(keywords, "show_prefix");
            if (resize_value == Py_True) {
                printPreffix = 1;
            } 
            else if (resize_value == Py_False) {
                printPreffix = 0;
            } 
            else {
                printPreffix = 1;
            }
        } 
        else {
            printPreffix = 1;
        }
    }

    if (PyArg_ParseTuple(args, "s", &string)) {
        if (printPreffix == 1) {
            post("[Python]: %s", string);
        } 
        else {
            post("%s", string);
        }
        PyErr_Clear();
    } 
    else if (PyArg_ParseTuple(args, "f", &string)) {
        if (printPreffix == 1) {
            post("[Python]: %f", string);
        } 
        else {
            post("%f", string);
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.print work with string or number.");  // Colocar melhor descrição do
        return NULL;
    }
    return PyLong_FromLong(0);
}

// =================================
PyObject *pderror(PyObject *self, PyObject *args) {
    (void)self;
    char *string;

    if (PyArg_ParseTuple(args, "s", &string)) {
        pd_error(NULL, "[Python]: %s", string);
        PyErr_Clear();
    } else {
        PyErr_SetString(
            PyExc_TypeError,
            "[Python] argument of pd.error must be a string");  
        return NULL;
    }
    return PyLong_FromLong(0);
}

// =================================
PyObject *pdsend(PyObject *self, PyObject *args) {
    (void)self;
    char *receiver;
    char *string;
    float floatNumber;
    int intNumber;
    PyObject *listargs;

    if (PyArg_ParseTuple(args, "ss", &receiver, &string)) {
        t_symbol *symbol = gensym(receiver);
        if (symbol->s_thing) {
            pd_symbol(symbol->s_thing, gensym(string));
        } else {
            post("[Python] pd.send not found object [r %s] in pd patch",
                 receiver);
        }
    } else if (PyArg_ParseTuple(args, "sf", &receiver, &floatNumber)) {
        t_symbol *symbol = gensym(receiver);
        if (symbol->s_thing) {
            pd_float(symbol->s_thing, floatNumber);
        } else {
            post("[Python] pd.send not found object [r %s] in pd patch",
                 receiver);
        }
    } else if (PyArg_ParseTuple(args, "si", &receiver, &intNumber)) {
        t_symbol *symbol = gensym(receiver);
        if (symbol->s_thing) {
            pd_float(symbol->s_thing, intNumber);
        } else {
            post("[Python] pd.send not found object [r %s] in pd patch",
                 receiver);
        }
    } else if (PyArg_ParseTuple(args, "sO", &receiver, &listargs)) {
        if (PyDict_Check(listargs)) {
            char error_message[100];
            sprintf(error_message,
                    "[Python] pd.send received a type 'dict', it must be a list, "
                    "string, int, or float.");
            PyErr_SetString(PyExc_TypeError,
                            error_message);  
            return NULL;
        }
        t_atom *list_array;
        int list_size = PyList_Size(listargs);
        list_array = (t_atom *)malloc(list_size * sizeof(t_atom));
        int i;
        for (i = 0; i < list_size; ++i) {
            PyObject *pValue_i = PyList_GetItem(listargs, i);
            if (PyLong_Check(pValue_i)) {  
                long result = PyLong_AsLong(pValue_i);
                float result_float = (float)result;
                list_array[i].a_type = A_FLOAT;
                list_array[i].a_w.w_float = result_float;
            } else if (PyFloat_Check(pValue_i)) {  
                                                  
                double result = PyFloat_AsDouble(pValue_i);
                float result_float = (float)result;
                list_array[i].a_type = A_FLOAT;
                list_array[i].a_w.w_float = result_float;
            } else if (PyUnicode_Check(pValue_i)) { 
                const char *result = PyUnicode_AsUTF8(pValue_i);
                list_array[i].a_type = A_SYMBOL;
                list_array[i].a_w.w_symbol = gensym(result);
            } else if (Py_IsNone(pValue_i)) {
                // post("None");
            } else {
                char error_message[100];
                sprintf(error_message,
                        "[Python] received a type '%s' in index %d of the "
                        "list, it must be a string, int, or float.",
                        pValue_i->ob_type->tp_name, i);
                PyErr_SetString(PyExc_TypeError,
                                error_message);  
                Py_DECREF(pValue_i);
                Py_DECREF(args);
                free(list_array);
                return NULL;
            }
        }
        if (gensym(receiver)->s_thing) {
            pd_list(gensym(receiver)->s_thing, &s_list, list_size, list_array);
        } else {
            post("[Python] object [r %s] not found", receiver);
        }
    } else {
        char error_message[100];
        PyObject *pValue_i = PyTuple_GetItem(args, 1);
        sprintf(error_message,
                "[Python] pd.send received a type '%s', it must be a "
                "string, int, or float.",
                pValue_i->ob_type->tp_name);
        PyErr_SetString(PyExc_TypeError, error_message);
        return NULL;
    }
    PyErr_Clear();
    return PyLong_FromLong(0);
}

// =================================
PyObject *pdtabwrite(PyObject *self, PyObject *args, PyObject *keywords) {
    (void)self;
    int resize = 0;
    int vecsize;
    t_garray *pdarray;
    t_word *vec;
    char *string;
    PyObject *PYarray;

    if (keywords == NULL) {
        resize = 0;
        PyErr_Clear();
    } 
    else {
        resize = PyDict_Contains(keywords, PyUnicode_FromString("resize"));
        if (resize == -1) {
            post("error");
        } 
        else if (resize == 1) {
            PyObject *resize_value = PyDict_GetItemString(keywords, "resize");
            if (resize_value == Py_True) {
                resize = 1;
            } 
            else if (resize_value == Py_False) {
                resize = 0;
            } 
            else {
                resize = 0;
            }
        } 
        else {
            resize = 0;
        }
    }

    // ================================
    PyObject *pd_module = PyImport_ImportModule("__main__");
    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, "py4pd");
    t_py *py4pd = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
    // ================================

    if (PyArg_ParseTuple(args, "sO", &string, &PYarray)) {
        t_symbol *pd_symbol = gensym(string);
        if (!(pdarray = (t_garray *)pd_findbyclass(pd_symbol, garray_class)))
            pd_error(py4pd, "[Python] Array %s not found.", string);
        else if (!garray_getfloatwords(pdarray, &vecsize, &vec))
            pd_error(py4pd, "[Python] Bad template for tabwrite '%s'.",
                     string);
        else {
            int i;
            if (resize == 1) {
                garray_resize_long(pdarray, PyList_Size(PYarray));
                vecsize = PyList_Size(PYarray);
                garray_getfloatwords(pdarray, &vecsize, &vec);
            }
            for (i = 0; i < vecsize; i++) {
                double result = PyFloat_AsDouble(PyList_GetItem(PYarray, i));
                float result_float = (float)result;
                vec[i].w_float = result_float;
            }
            garray_redraw(pdarray);
            PyErr_Clear();
        }
    }
    return PyLong_FromLong(0);
}

// =================================
PyObject *pdtabread(PyObject *self, PyObject *args) {
    (void)self;
    int vecsize;
    t_garray *pdarray;
    t_word *vec;
    char *string;

    // ================================
    PyObject *pd_module = PyImport_ImportModule("__main__");
    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, "py4pd");
    t_py *py4pd = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
    // ================================

    if (PyArg_ParseTuple(args, "s", &string)) {
        t_symbol *pd_symbol = gensym(string);
        if (!(pdarray = (t_garray *)pd_findbyclass(pd_symbol, garray_class))) {
            pd_error(py4pd, "[Python] Array %s not found.", string);
            PyErr_SetString(PyExc_TypeError,
                            "[Python] pd.tabread: array not found");
        } else {
            int i;
            garray_getfloatwords(pdarray, &vecsize, &vec);
            PyObject *list = PyList_New(vecsize);
            for (i = 0; i < vecsize; i++) {
                PyList_SetItem(list, i, PyFloat_FromDouble(vec[i].w_float));
            }
            PyErr_Clear();
            return list;
        }
    } else {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.tabread: wrong arguments");
        return NULL;
    }
    return NULL;
}

// =================================
PyObject *pdhome(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *pd_module = PyImport_ImportModule("__main__");
    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, "py4pd");
    t_py *py4pd = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");

    // check if there is no argument
    if (!PyArg_ParseTuple(args, "")) {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.home: no argument expected");
        return NULL;
    }
    return PyUnicode_FromString(py4pd->home_path->s_name);
}


// =================================
PyObject *pdtempfolder(PyObject *self, PyObject *args) {
    (void)self;
    if (!PyArg_ParseTuple(args, "")) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.samplerate: no argument expected");
        return NULL;
    }
    PyObject *pd_module = PyImport_ImportModule("__main__");
    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, "py4pd");
    t_py *py4pd = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
    py4pd_tempfolder(py4pd);
    return PyUnicode_FromString(py4pd->temp_folder->s_name);
}


// =================================
PyObject *pdshowimage(PyObject *self, PyObject *args) {
    (void)self;
    char *string;

    PyObject *pd_module = PyImport_ImportModule("__main__");
    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, "py4pd");
    t_py *py4pd = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");

    PY4PD_erase(py4pd, py4pd->x_glist);
    if (PyArg_ParseTuple(args, "s", &string)) {
        t_symbol *filename = gensym(string);
        if (py4pd->x_def_img) {
            py4pd->x_def_img = 0;
        }
        FILE *file;
        char *ext = strrchr(filename->s_name, '.');
        if (strcmp(ext, ".ppm") == 0) {
            char magic_number[3];
            int width, height, max_color_value;
            file = fopen(filename->s_name, "r");
            fscanf(file, "%s\n%d %d\n%d\n", magic_number, &width, &height, &max_color_value);
            py4pd->x_width = width;
            py4pd->x_height = height;
            fclose(file);
        }
        else if (strcmp(ext, ".gif") == 0){
            file = fopen(filename->s_name, "rb");
            fseek(file, 6, SEEK_SET);
            fread(&py4pd->x_width, 2, 1, file);
            fread(&py4pd->x_height, 2, 1, file);
            fclose(file);
        }
        else if (strcmp(ext, ".png") == 0) {
            file = fopen(filename->s_name, "rb");
            int width, height;
            fseek(file, 16, SEEK_SET);
            fread(&width, 4, 1, file);
            fread(&height, 4, 1, file);
            width = py4pd_ntohl(width);
            height = py4pd_ntohl(height);
            py4pd->x_width = width;
            py4pd->x_height = height;
        }

        else{
            pd_error(py4pd, "[Python] pd.showimage: file format not supported");
            PyErr_SetString(PyExc_TypeError, "[Python] pd.showimage: file format not supported");
            return NULL;
        }

        if (glist_isvisible(py4pd->x_glist) && gobj_shouldvis((t_gobj *)py4pd, py4pd->x_glist)) {
            const char *file_name_open = PY4PD_filepath(py4pd, filename->s_name);
            if (file_name_open) {
                py4pd->x_filename = filename;
                py4pd->x_fullname = gensym(file_name_open);

                if (py4pd->x_def_img) {
                    py4pd->x_def_img = 0;
                }

                if (glist_isvisible(py4pd->x_glist) && gobj_shouldvis((t_gobj *)py4pd, py4pd->x_glist)) {
                    PY4PD_erase(py4pd, py4pd->x_glist);
                    sys_vgui(
                        "if {[info exists %lx_picname] == 0} {image create "
                        "photo %lx_picname -file \"%s\"\n set %lx_picname "
                        "1\n}\n",
                        py4pd->x_fullname, py4pd->x_fullname, file_name_open,
                        py4pd->x_fullname);
                    PY4PD_draw(py4pd, py4pd->x_glist, 1);
                }
                PY4PD_draw_io_let(py4pd);
            } 
            else {
                pd_error(py4pd, "[Python]: Error displaying image, file not found");
                PyErr_Clear();
                Py_RETURN_NONE;
            }
        } 
        else {
            pd_error(py4pd, "[Python]: Error displaying image");
            PyErr_Clear();
            Py_RETURN_NONE;
        }

    } 
    else {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.showimage received wrong arguments");
         PyErr_Clear();
        Py_RETURN_NONE;
    }
    // python return True
    PyErr_Clear();
    Py_RETURN_TRUE;
}

// =================================
// ========== AUDIO CONFIG =========
// =================================

PyObject *pdsamplerate(PyObject *self, PyObject *args) {
    (void)self;
    if (!PyArg_ParseTuple(args, "")) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.samplerate: no argument expected");
        return NULL;
    }
    t_sample sr = sys_getsr();
    return PyLong_FromLong(sr);
}

// =================================
PyObject *pdveczise(PyObject *self, PyObject *args) {
    (void)self;
    if (!PyArg_ParseTuple(args, "")) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.samplerate: no argument expected");
        return NULL;
    }
    t_sample vector = sys_getblksize();
    return PyLong_FromLong(vector);
}

// =================================
PyObject *pdkey(PyObject *self, PyObject *args) {
    //get values from Dict salved in x->param
    (void)self;
    char *key;
    if (!PyArg_ParseTuple(args, "s", &key)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.key: no argument expected");
        return NULL;
    }
    // ================================
    PyObject *pd_module = PyImport_ImportModule("__main__");
    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, "py4pd");
    t_py *py4pd = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
    // ================================
    // get value from dict
    if (py4pd->Dict == NULL) {
        PyErr_Clear();
        Py_RETURN_NONE;
    }

    PyObject *value = PyDict_GetItemString(py4pd->Dict, key);
    if (value == NULL) {
        PyErr_Clear();
        Py_RETURN_NONE;
    }
    Py_INCREF(value);
    return value;
}

// =================================
PyObject *pdmoduleError;

// =================================
PyMethodDef PdMethods[] = {

    // PureData inside Python
    {"out", (PyCFunction)pdout, METH_VARARGS | METH_KEYWORDS, "Output in out0 from PureData"},
    {"send", pdsend, METH_VARARGS, "Send message to PureData, it can be received with the object [receive]"},
    {"print", (PyCFunction)pdprint, METH_VARARGS | METH_KEYWORDS,  "Print informations in PureData Console"},
    {"error", pderror, METH_VARARGS, "Print informations in error format (red) in PureData Console"},
    {"tabwrite", (PyCFunction)pdtabwrite, METH_VARARGS | METH_KEYWORDS, "Write data to PureData tables/arrays"},
    {"tabread", pdtabread, METH_VARARGS, "Read data from PureData tables/arrays"},
    
    // Pic
    {"show", pdshowimage, METH_VARARGS, "Show image in PureData, it must be .gif, .bmp, .ppm"},

    // Files
    {"home", pdhome, METH_VARARGS, "Get PureData Patch Path Folder"},
    {"tempfolder", pdtempfolder, METH_VARARGS, "Get PureData Temp Folder"},

    // User
    {"getkey", pdkey, METH_VARARGS, "Get Object User Parameters"},

    // audio
    {"samplerate", pdsamplerate, METH_VARARGS, "Get PureData SampleRate"},
    {"vecsize", pdveczise, METH_VARARGS, "Get PureData Vector Size"},


    // library methods
    {"addobject", (PyCFunction)pdAddPyObject, METH_VARARGS | METH_KEYWORDS, "It add python functions as objects"},


    {NULL, NULL, 0, NULL}  //
};

// =================================
struct PyModuleDef pdmodule = {
    PyModuleDef_HEAD_INIT,
    "pd", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module, or -1 if the module
             keeps state in global variables. */
    PdMethods,  // Methods
    NULL,
    NULL,
    NULL,
    NULL,
};

// =================================

PyMODINIT_FUNC PyInit_pd() {
    PyObject *py4pdmodule;
    
    py4pdmodule = PyModule_Create(&pdmodule);
    
    if (py4pdmodule == NULL) {
        return NULL;
    }
    
    PyObject *puredata_samplerate, *puredata_vecsize;

    puredata_samplerate = PyLong_FromLong(sys_getsr());
    puredata_vecsize = PyLong_FromLong(sys_getblksize());

    PyModule_AddObject(py4pdmodule, "SAMPLERATE", puredata_samplerate);
    PyModule_AddObject(py4pdmodule, "VECSIZE", puredata_vecsize);

    pdmoduleError = PyErr_NewException("spam.error", NULL, NULL);
    
    Py_XINCREF(pdmoduleError);

    if (PyModule_AddObject(py4pdmodule, "moduleerror", pdmoduleError) < 0) {
        Py_XDECREF(pdmoduleError);
        Py_CLEAR(pdmoduleError);
        Py_DECREF(py4pdmodule);
        return NULL;
    }
    return py4pdmodule;
}
