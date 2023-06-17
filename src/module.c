#include "py4pd.h"
#include "utils.h"
#include "pic.h"  
#include "ext-libraries.h"

#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#include <numpy/arrayobject.h>


// ======================================
// ======== py4pd embbeded module =======
// ======================================

static t_py *get_py4pd_object(void){
    PyObject *pd_module = PyImport_ImportModule("pd");
    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, "py4pd");
    if (py4pd_capsule == NULL){
        post("[Python] py4pd capsule not found.");
        return NULL;
    }
    t_py *py4pd = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
    return py4pd;
}

// ======================================
// ======== py4pd embbeded module =======
// ======================================

PyObject *pdout(PyObject *self, PyObject *args, PyObject *keywords){
    (void)keywords;
    (void)self;
    
    t_py *py4pd = get_py4pd_object();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }

    if (keywords != NULL) { // special case for py.iterate
        PyObject *pyiterate = PyDict_GetItemString(keywords, "pyiterate"); // it gets the data type output
        PyObject *pycollect = PyDict_GetItemString(keywords, "pycollect"); // it gets the data type output
        PyObject *pyout = PyDict_GetItemString(keywords, "pyout"); // it gets the data type output
        if (pyiterate == Py_True || pycollect == Py_True || pyout == Py_True) {
            py4pd->outPyPointer = 1;
            PyObject *copyModule = PyImport_ImportModule("copy");
            PyObject *deepcopyFunction = PyObject_GetAttrString(copyModule, "deepcopy");
            PyObject *argsTuple = PyTuple_Pack(1, args);
            PyObject *newObject = PyObject_CallObject(deepcopyFunction, argsTuple);
            PyObject *element = PyTuple_GetItem(newObject, 0);
            py4pd_convert_to_pd(py4pd, element);
            py4pd->outPyPointer = 0;
            Py_DECREF(copyModule);
            Py_DECREF(deepcopyFunction);
            Py_DECREF(argsTuple);
            Py_DECREF(newObject);
            Py_DECREF(element);
            return Py_True;
        }
    }
    py4pd_convert_to_pd(py4pd, args);
    return PyLong_FromLong(0);
}

// =================================
PyObject *pdprint(PyObject *self, PyObject *args, PyObject *keywords) {
    (void)self;
    int printPrefix = 1;
    int objPrefix = 1;

    t_py *py4pd = get_py4pd_object();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }


    if (keywords == NULL) {
        printPrefix = 1;
        PyErr_Clear();
    } 
    else {
        printPrefix = PyDict_Contains(keywords, PyUnicode_FromString("show_prefix"));
        objPrefix = PyDict_Contains(keywords, PyUnicode_FromString("obj_prefix"));
        if (printPrefix == -1) {
            pd_error(NULL, "[Python] pd.print: error in show_prefix argument.");
        } 
        else if (printPrefix == 1) {
            PyObject *resize_value = PyDict_GetItemString(keywords, "show_prefix");
            if (resize_value == Py_True) {
                printPrefix = 1;
            } 
            else if (resize_value == Py_False) {
                printPrefix = 0;
            } 
            else {
                post("[Python] pd.print: show_prefix argument must be True or False.");
                printPrefix = 1;
            }
        } 
        else {
            printPrefix = 1;
        }
        // check if obj_prefix is present and see if it is True or False
        if (objPrefix == -1) {
            pd_error(NULL, "[Python] pd.print: error in obj_prefix argument.");
        } 
        else if (objPrefix == 1) {
            PyObject *resize_value = PyDict_GetItemString(keywords, "obj_prefix");
            if (resize_value == Py_True) {
                objPrefix = 1;
            } 
            else if (resize_value == Py_False) {
                objPrefix = 0;
            } 
            else {
                post("[Python] pd.print: obj_prefix argument must be True or False.");
                objPrefix = 0;
            }
        } 
        else {
            objPrefix = 0;
        }
    }

    PyObject* obj;
    if (PyArg_ParseTuple(args, "O", &obj)) {
        PyObject* str = PyObject_Str(obj);
        if (str == NULL) {
            PyErr_SetString(PyExc_TypeError, "[Python] pd.print failed to convert object to string.");
            return NULL;
        }
        const char* str_value = PyUnicode_AsUTF8(str);
        if (str_value == NULL) {
            PyErr_SetString(PyExc_TypeError, "[Python] pd.print failed to convert string object to UTF-8.");
            Py_DECREF(str);
            return NULL;
        }
        if (printPrefix == 1) { //
            if (py4pd->objectName == NULL){
                post("[Python]: %s", str_value);
            }
            else{
                post("[%s]: %s", py4pd->objectName->s_name, str_value);
            }
            return PyLong_FromLong(0);
        } 
        else {
            post("%s", str_value);
        }
    } 
    else {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.print works with strings, numbers, and any other valid Python object.");
        return NULL;
    }

    return PyLong_FromLong(0);
}

// =================================
PyObject *pdlogpost(PyObject *self, PyObject *args) {
    (void)self;
    int postlevel;
    char *string;

    if (PyArg_ParseTuple(args, "is", &postlevel, &string)) {
        logpost(NULL, postlevel, "%s", string);
        return PyLong_FromLong(0);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.logpost works with strings and numbers.");
        return NULL;
    }
}


// =================================
PyObject *pderror(PyObject *self, PyObject *args) {
    (void)self;

    char *string;

    t_py *py4pd = get_py4pd_object();

    if (PyArg_ParseTuple(args, "s", &string)) {
        if (py4pd == NULL){
            pd_error(NULL, "%s", string);
            PyErr_Clear();
            return PyLong_FromLong(0);
        }

        if (py4pd->pyObject == 1){
            pd_error(py4pd, "[%s]: %s", py4pd->objectName->s_name, string);
        }
        else{
            if (py4pd->function_name == NULL){
                pd_error(py4pd, "%s", string);
            }
            else{
                pd_error(py4pd, "[%s]: %s", py4pd->function_name->s_name, string);
            }
        }
        PyErr_Clear();
    } 
    else {
        PyErr_SetString(
            PyExc_TypeError,
            "[Python] argument of pd.error must be a string");  
        return NULL;
    }
    return PyLong_FromLong(0);
}

// =================================
PyObject *pipinstall(PyObject *self, PyObject *args){
    (void)self;
    char *package;

    t_py *py4pd = get_py4pd_object();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }

    if (PyArg_ParseTuple(args, "s", &package)) {
        t_atom argv[2];
        SETSYMBOL(argv, gensym("py4pd"));
        SETSYMBOL(argv+1, gensym("pipinstall"));
        
        char *pyScripts_folder = malloc(strlen(py4pd->py4pdPath->s_name) + 20); // allocate extra space
        snprintf(pyScripts_folder, strlen(py4pd->py4pdPath->s_name) + 20, "%s/resources/scripts", py4pd->py4pdPath->s_name);


        PyObject *sys_path = PySys_GetObject("path");
        PyList_Append(sys_path, PyUnicode_FromString(pyScripts_folder));
        PyObject *pipinstall_module = PyImport_ImportModule("py4pd");
        if (pipinstall_module == NULL) {
            PyErr_Print();
            return NULL;
        }
        PyObject *pipinstall_function = PyObject_GetAttrString(pipinstall_module, "pipinstall");
        if (pipinstall_function == NULL) {
            PyErr_Print();
            return NULL;
        }
        // create new List
        PyObject *argsList = PyList_New(2);
        PyList_SetItem(argsList, 0, PyUnicode_FromString("global"));
        PyList_SetItem(argsList, 1, PyUnicode_FromString(package));
        PyObject *result = PyObject_CallObject(pipinstall_function, argsList);
        if (result == NULL) {
            PyErr_Print();
            return NULL;
        }
        Py_DECREF(pipinstall_module);
        Py_DECREF(pipinstall_function);
        Py_DECREF(argsList);
        Py_DECREF(result);
        free(pyScripts_folder);
        return PyLong_FromLong(0);
    }
    else {
        PyErr_SetString(
            PyExc_TypeError,
            "[Python] argument of pd.pipinstall must be a string");  
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
        } 
        else {
            post("[Python] pd.send not found object [r %s] in pd patch",
                 receiver);
        }
    } 
    else if (PyArg_ParseTuple(args, "sf", &receiver, &floatNumber)) {
        t_symbol *symbol = gensym(receiver);
        if (symbol->s_thing) {
            pd_float(symbol->s_thing, floatNumber);
        } 
        else {
            post("[Python] pd.send not found object [r %s] in pd patch",
                 receiver);
        }
    } else if (PyArg_ParseTuple(args, "si", &receiver, &intNumber)) {
        t_symbol *symbol = gensym(receiver);
        if (symbol->s_thing) {
            pd_float(symbol->s_thing, intNumber);
        } 
        else {
            post("[Python] pd.send not found object [r %s] in pd patch",
                 receiver);
        }
    } 
    else if (PyArg_ParseTuple(args, "sO", &receiver, &listargs)) {
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
            } 
            else if (PyFloat_Check(pValue_i)) {  
                                                  
                double result = PyFloat_AsDouble(pValue_i);
                float result_float = (float)result;
                list_array[i].a_type = A_FLOAT;
                list_array[i].a_w.w_float = result_float;
            } 
            else if (PyUnicode_Check(pValue_i)) { 
                const char *result = PyUnicode_AsUTF8(pValue_i);
                list_array[i].a_type = A_SYMBOL;
                list_array[i].a_w.w_symbol = gensym(result);
            } 
            else if (Py_IsNone(pValue_i)) {
                // post("None");
            }
            else {
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
            Py_DECREF(pValue_i);
        }
        if (gensym(receiver)->s_thing) {
            pd_list(gensym(receiver)->s_thing, &s_list, list_size, list_array);
        } 
        else {
            pd_error(NULL, "[Python] object [r %s] not found", receiver);
        }
    } 
    else {
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
    t_py *py4pd = get_py4pd_object();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }

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
PyObject *pdtabread(PyObject *self, PyObject *args, PyObject *keywords) {
    (void)self;
    int vecsize;
    t_garray *pdarray;
    t_word *vec;
    char *string;
    int numpy;

    // ================================
    PyObject *pd_module = PyImport_ImportModule("pd");
    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, "py4pd");
    t_py *py4pd = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
    // ================================

    if (keywords == NULL) {
        numpy = 0;
        PyErr_Clear();
    } 
    else{
        numpy = PyDict_Contains(keywords, PyUnicode_FromString("numpy"));
        if (numpy == -1) {
            pd_error(NULL, "[Python] Check the keyword arguments.");
            return NULL;
        } 
        else if (numpy == 1) {
            PyObject *numpy_value = PyDict_GetItemString(keywords, "numpy");
            if (numpy_value == Py_True) {
                numpy = 1;
                int numpyArrayImported = _import_array();
                if (numpyArrayImported == 0) {
                    py4pd->numpyImported = 1;
                }
                else{
                    py4pd->numpyImported = 0;
                    pd_error(py4pd, "[py4pd] Not possible to import numpy array");
                    return NULL;
                }
            } 
            else if (numpy_value == Py_False) {
                numpy = 0;
            } 
            else {
                numpy = 0;
            }
        } 
        else {
            numpy = 0;
        }
    }



    if (PyArg_ParseTuple(args, "s", &string)) {
        t_symbol *pd_symbol = gensym(string);
        if (!(pdarray = (t_garray *)pd_findbyclass(pd_symbol, garray_class))) {
            pd_error(py4pd, "[Python] Array %s not found.", string);
            PyErr_SetString(PyExc_TypeError,
                            "[Python] pd.tabread: array not found");
        } 
        else {
            int i;
            if (numpy == 0) {
                garray_getfloatwords(pdarray, &vecsize, &vec);
                PyObject *list = PyList_New(vecsize);
                for (i = 0; i < vecsize; i++) {
                    PyList_SetItem(list, i, PyFloat_FromDouble(vec[i].w_float));
                }
                PyErr_Clear();
                return list;
            }
            else if (numpy == 1) {
                garray_getfloatwords(pdarray, &vecsize, &vec);
                const npy_intp dims = vecsize;
                // send double float array to numpy
                PyObject *array = PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT, vec);
                PyErr_Clear();
                return array;

            }
            else {
                pd_error(py4pd, "[Python] Check the keyword arguments.");
                return NULL;
            }
        }
    } 
    else {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.tabread: wrong arguments");
        return NULL;
    }
    return NULL;
}

// =================================
PyObject *pdhome(PyObject *self, PyObject *args) {
    (void)self;

    t_py *py4pd = get_py4pd_object();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }

    // check if there is no argument
    if (!PyArg_ParseTuple(args, "")) {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.home: no argument expected");
        return NULL;
    }
    return PyUnicode_FromString(py4pd->pdPatchFolder->s_name);
}

// =================================
PyObject *py4pdfolder(PyObject *self, PyObject *args) {
    (void)self;

    t_py *py4pd = get_py4pd_object();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }

    // check if there is no argument
    if (!PyArg_ParseTuple(args, "")) {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.py4pdfolder: no argument expected");
        return NULL;
    }
    return PyUnicode_FromString(py4pd->py4pdPath->s_name);
}


// =================================
PyObject *pdtempfolder(PyObject *self, PyObject *args) {
    (void)self;
    if (!PyArg_ParseTuple(args, "")) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.samplerate: no argument expected");
        return NULL;
    }
    t_py *py4pd = get_py4pd_object();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }
    createPy4pdTempFolder(py4pd);
    return PyUnicode_FromString(py4pd->tempPath->s_name);
}

// =================================
PyObject *pdshowimage(PyObject *self, PyObject *args) {
    (void)self;
    char *string;

    t_py *py4pd = get_py4pd_object();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }

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
        pd_error(py4pd, "[Python] pd.showimage received wrong arguments");
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

static PyObject* pdsamplerate(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;

    float sr = sys_getsr(); // call PureData's sys_getsr function to get the current sample rate
    return PyFloat_FromDouble(sr); // return the sample rate as a Python float object
}

// =================================
PyObject *pdveczise(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    t_py *py4pd = get_py4pd_object();
    t_sample vector;
    vector = py4pd->vectorSize;

    return PyLong_FromLong(vector);
}


// =================================
// ========== Utilities ============
// =================================

PyObject *pdkey(PyObject *self, PyObject *args) {
    //get values from Dict salved in x->param
    (void)self;
    char *key;
    if (!PyArg_ParseTuple(args, "s", &key)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.key: no argument expected");
        return NULL;
    }
    
    t_py *py4pd = get_py4pd_object();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }

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
PyObject *pditerate(PyObject *self, PyObject *args){
    (void)self;

    PyObject *iter, *item;

    t_py *py4pd = get_py4pd_object();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }

    if (!PyTuple_Check(args)) {
        PyErr_SetString(PyExc_TypeError, "pditerate() argument must be a tuple");
        return NULL;
    }
    iter = PyObject_GetIter(args);
    if (iter == NULL) {
        PyErr_SetString(PyExc_TypeError, "pditerate() argument must be iterable");
        return NULL;
    }
    while ((item = PyIter_Next(iter))) {
        if (!PyList_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "pditerate() argument must be a list");
            return NULL;
        }
        // get item size
        int size = PyList_Size(item);
        for (int i = 0; i < size; i++) {
            PyObject *out_args = PyList_GetItem(item, i);
            void *pData = pyobject_to_pointer(out_args);
            if (Py_REFCNT(out_args) == 1) { // TODO: think about how will clear this
                Py_INCREF(out_args);
            }
            t_atom pointer_atom;
            SETPOINTER(&pointer_atom, pData);
            outlet_anything(py4pd->x_obj.ob_outlet, gensym("PyObject"), 1, &pointer_atom);
        }
    }
    Py_DECREF(iter);
    Py_RETURN_NONE;
}

// =================================
PyObject *getobjpointer(PyObject *self, PyObject *args){
    (void)self;
    (void)args;
    
    t_py *py4pd = get_py4pd_object();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }
    
    return PyUnicode_FromFormat("%p", py4pd);
}

// =================================
PyObject *setglobalvar(PyObject *self, PyObject *args){
    (void)self;

    PyObject* globalsDict = PyEval_GetGlobals();
    t_py *py4pd = get_py4pd_object();
    char varString[MAXPDSTRING];

    char *varName;
    PyObject *value;
    if (!PyArg_ParseTuple(args, "sO", &varName, &value)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.setglobalvar: wrong arguments");
        return NULL;
    }
    snprintf(varString, MAXPDSTRING, "%s_%p", varName, py4pd);
    PyObject* globalVariableString = PyUnicode_FromFormat(varString);
    PyObject* globalValue = PyDict_GetItem(globalsDict, globalVariableString);
    if (globalValue == NULL) {
        PyDict_SetItem(globalsDict, globalVariableString, value);
    } 
    else {
        Py_DECREF(globalValue);
        PyDict_SetItem(globalsDict, globalVariableString, value);
    }
    Py_DECREF(globalVariableString);
    Py_RETURN_TRUE;
}

// =================================
PyObject *getglobalvar(PyObject *self, PyObject *args, PyObject *keywords){
    (void)self;

    PyObject* globalsDict = PyEval_GetGlobals();
    if (globalsDict == NULL) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.getglobalvar: globalsDict is NULL");
        return NULL;
    }


    t_py *py4pd = get_py4pd_object();
    char varString[MAXPDSTRING];

    char *varName;
    if (!PyArg_ParseTuple(args, "s", &varName)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.setglobalvar: wrong arguments");
        return NULL;
    }

    PyObject *initial_value;
    if (keywords != NULL) {
        initial_value = PyDict_GetItemString(keywords, "initial_value");
    }
    else {
        initial_value = Py_None;
    }
    snprintf(varString, MAXPDSTRING, "%s_%p", varName, py4pd);
    PyObject* globalValue = PyDict_GetItemString(globalsDict, varString);
    if (globalValue == NULL) {
        PyDict_SetItemString(globalsDict, varString, initial_value);
        globalValue = PyDict_GetItemString(globalsDict, varString);
        Py_INCREF(globalValue);
    } 
    else {
        Py_INCREF(globalValue);
    }
    return Py_BuildValue("O", globalValue);
}

// =================================
PyObject *pdmoduleError;

// =================================
PyMethodDef PdMethods[] = {

    // PureData inside Python
    {"out", (PyCFunction)pdout, METH_VARARGS | METH_KEYWORDS, "Output in out0 from PureData"},
    {"send", pdsend, METH_VARARGS, "Send message to PureData, it can be received with the object [receive]"},
    {"print", (PyCFunction)pdprint, METH_VARARGS | METH_KEYWORDS,  "Print informations in PureData Console"},
    {"logpost", pdlogpost, METH_VARARGS, "Print informations in PureData Console"},
    {"error", pderror, METH_VARARGS, "Print informations in error format (red) in PureData Console"},
    {"tabwrite", (PyCFunction)pdtabwrite, METH_VARARGS | METH_KEYWORDS, "Write data to PureData tables/arrays"},
    {"tabread", (PyCFunction)pdtabread, METH_VARARGS | METH_KEYWORDS, "Read data from PureData tables/arrays"},
    
    // Pic
    {"show", pdshowimage, METH_VARARGS, "Show image in PureData, it must be .gif, .bmp, .ppm"},

    // Files
    {"home", pdhome, METH_VARARGS, "Get PureData Patch Path Folder"},
    {"py4pdfolder", py4pdfolder, METH_VARARGS, "Get PureData Py4PD Folder"},
    {"tempfolder", pdtempfolder, METH_VARARGS, "Get PureData Temp Folder"},

    // User
    {"getkey", pdkey, METH_VARARGS, "Get Object User Parameters"},

    // audio
    {"samplerate", pdsamplerate, METH_NOARGS, "Get PureData SampleRate"},
    {"vecsize", pdveczise, METH_NOARGS, "Get PureData Vector Size"},

    // library methods
    {"addobject", (PyCFunction)pdAddPyObject, METH_VARARGS | METH_KEYWORDS, "It adds python functions as objects"},

    // OpenMusic Methods
    {"iterate", pditerate, METH_VARARGS, "It iterates throw one list of PyObjects"},

    // Others
    {"getobjpointer", getobjpointer, METH_NOARGS, "Get PureData Object Pointer"},
    {"getstrpointer", getobjpointer, METH_NOARGS, "Get PureData Object Pointer"},
    {"setglobalvar", setglobalvar, METH_VARARGS, "It sets a global variable for the Object, it is not clear after the execution of the function"},
    {"getglobalvar", (PyCFunction)getglobalvar, METH_VARARGS | METH_KEYWORDS, "It gets a global variable for the Object, it is not clear after the execution of the function"},

    // pip
    // {"pip", (PyCFunction)pipinstall, METH_VARARGS, "It installs python packages"},

    {NULL, NULL, 0, NULL}  //
};

// =================================
struct PyModuleDef pdmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pd", /* name of module */
    .m_doc = "pd module provide function to interact with PureData, see the docs in www.charlesneimog.com/py4pd",
    .m_size = 0,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    .m_methods = PdMethods,  // Methods of the module
    .m_slots = NULL, /* m_slots, that is the slots for multi-phase initialization */
    .m_traverse = NULL, /* m_traverse, that is the traverse function for GC */
    .m_clear = NULL, /* m_free, that is the free function for GC */
};

// =================================

PyMODINIT_FUNC PyInit_pd() {
    PyObject *py4pdmodule;
    
    py4pdmodule = PyModule_Create(&pdmodule);
    
    if (py4pdmodule == NULL) {
        return NULL;
    }
    
    PyObject *puredata_samplerate, *puredata_vecsize, *visObject, *audioINObject, *audioOUTObject, *pdAudio;

    puredata_samplerate = PyLong_FromLong(sys_getsr());
    puredata_vecsize = PyLong_FromLong(sys_getblksize());
    
    visObject = PyUnicode_FromString("VIS");
    audioINObject = PyUnicode_FromString("AUDIOIN");
    audioOUTObject = PyUnicode_FromString("AUDIOOUT");
    pdAudio = PyUnicode_FromString("AUDIO");

    PyModule_AddObject(py4pdmodule, "SAMPLERATE", puredata_samplerate);
    PyModule_AddObject(py4pdmodule, "VECSIZE", puredata_vecsize);
    PyModule_AddObject(py4pdmodule, "VIS", visObject);
    PyModule_AddObject(py4pdmodule, "AUDIOIN", audioINObject);
    PyModule_AddObject(py4pdmodule, "AUDIOOUT", audioOUTObject);
    PyModule_AddObject(py4pdmodule, "AUDIO", pdAudio);

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
