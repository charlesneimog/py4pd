#include "pylibraries.h"
#include "m_pd.h"
#include "py4pd.h"
#include "py4pd_utils.h"
#include "py4pd_pic.h"

t_class *pyNewObject_VIS;

static t_class *pyNewObject;
static t_class *pyNewObject_AudioIn;
static t_class *pyNewObject_AudioOut;
static t_class *pyNewObject_Audio;

static t_class *py4pdInlets_proxy_class;

// =====================================
void py4pdInlets_proxy_pointer(t_py4pdInlet_proxy *x, t_atom *argv){
    t_py *py4pd = (t_py *)x->p_master;
    PyObject *pValue;
    pValue = pointer_to_pyobject(argv);
    PyTuple_SetItem(py4pd->argsDict, x->inletIndex, pValue);
    return;
}


// =====================================
void py4pdInlets_proxy_anything(t_py4pdInlet_proxy *x, t_symbol *s, int ac, t_atom *av){
    t_py *py4pd = (t_py *)x->p_master;
    if (ac == 0){
        PyObject *pyInletValue = PyUnicode_FromString(s->s_name);
        PyTuple_SetItem(py4pd->argsDict, x->inletIndex, pyInletValue);
    }
    else{
        // create a list in Python
        PyObject *pyInletValue = PyList_New(ac + 1);
        if (isNumericOrDot(s->s_name)){
        }
        else{
        }
        PyList_SetItem(pyInletValue, 0, PyUnicode_FromString(s->s_name));
        for (int i = 0; i < ac; i++){
            if (av[i].a_type == A_FLOAT){ // TODO: check if it is an int or a float
                PyList_SetItem(pyInletValue, i + 1, PyLong_FromLong(av[i].a_w.w_float));
            }
            else if (av[i].a_type == A_SYMBOL){
                PyList_SetItem(pyInletValue, i + 1, PyUnicode_FromString(av[i].a_w.w_symbol->s_name));
            }
        }
        PyTuple_SetItem(py4pd->argsDict, x->inletIndex, pyInletValue);
    }
}

// =====================================
void py4pdInlets_proxy_list(t_py4pdInlet_proxy *x, t_symbol *s, int ac, t_atom *av){
    (void)s;
    t_py *py4pd = (t_py *)x->p_master;
    PyObject *pyInletValue;
    if (ac == 0){
        pyInletValue = PyUnicode_FromString(s->s_name);
        PyTuple_SetItem(py4pd->argsDict, 0, pyInletValue);
    }
    else if (ac == 1){
        if (av[0].a_type == A_FLOAT){
            int isInt = (int)av[0].a_w.w_float == av[0].a_w.w_float;
            if (isInt){
                PyTuple_SetItem(py4pd->argsDict, x->inletIndex, PyLong_FromLong(av[0].a_w.w_float));
            }
            else{
                PyTuple_SetItem(py4pd->argsDict, x->inletIndex, PyFloat_FromDouble(av[0].a_w.w_float));
            }
        }
        else if (av[0].a_type == A_SYMBOL){
            pyInletValue = PyUnicode_FromString(av[0].a_w.w_symbol->s_name);
            PyTuple_SetItem(py4pd->argsDict, x->inletIndex, pyInletValue);
        }
    }
    else { // NOTE: Need some work here
        pyInletValue = PyList_New(ac);
        for (int i = 0; i < ac; i++){
            if (av[i].a_type == A_FLOAT){ 
                int isInt = (int)av[i].a_w.w_float == av[i].a_w.w_float;
                if (isInt){
                    PyList_SetItem(pyInletValue, i, PyLong_FromLong(av[i].a_w.w_float));
                }
                else{
                    PyList_SetItem(pyInletValue, i, PyFloat_FromDouble(av[i].a_w.w_float));
                }
            }

            else if (av[i].a_type == A_SYMBOL){
                PyList_SetItem(pyInletValue, i, PyUnicode_FromString(av[i].a_w.w_symbol->s_name));
            }
        }
        PyTuple_SetItem(py4pd->argsDict, x->inletIndex, pyInletValue);
    }
    return;
}

// =====================================
void py_bang(t_py *x){
    // check if number of args is 0
    if (x->py_arg_numbers != 0){
        pd_error(x, "[py4pd] Python Objects just accept bangs when the Python Function has no arguments.");
        return;
    }
    PyObject *objectCapsule = py4pd_add_pd_object(x);
    if (objectCapsule == NULL){
        pd_error(x, "[py4pd] Failed to add object to Python");
        return;
    }
    PyObject *pArgs = PyTuple_New(0);
    PyObject *pValue = PyObject_CallObject(x->function, pArgs);
    if (pValue != NULL) { 
        py4pd_convert_to_pd(x, pValue); 
    }
    else{
        Py_XDECREF(pValue);
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[Python] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        PyErr_Clear();
    }
    return;
}

// =====================================
void py_anything(t_py *x, t_symbol *s, int ac, t_atom *av){
    if (s == gensym("bang")){
        py_bang(x);
        return;
    }
    PyObject *pyInletValue;
    PyObject *pValue;
    if (ac == 0){
        pyInletValue = PyUnicode_FromString(s->s_name);
        PyTuple_SetItem(x->argsDict, 0, pyInletValue);
    }
    else if ((s == gensym("list") || s == gensym("anything")) && ac > 0){
        pyInletValue = PyList_New(ac);
        for (int i = 0; i < ac; i++){
            if (av[i].a_type == A_FLOAT){ 
                int isInt = (int)av[i].a_w.w_float == av[i].a_w.w_float;
                if (isInt){
                    PyList_SetItem(pyInletValue, i, PyLong_FromLong(av[i].a_w.w_float));
                }
                else{
                    PyList_SetItem(pyInletValue, i, PyFloat_FromDouble(av[i].a_w.w_float));
                }
            }
            else if (av[i].a_type == A_SYMBOL){
                PyList_SetItem(pyInletValue, i, PyUnicode_FromString(av[i].a_w.w_symbol->s_name));
            }
        }
        PyTuple_SetItem(x->argsDict, 0, pyInletValue);
    }
    else if ((s == gensym("float") || s == gensym("symbol")) && ac == 1){
        if (av[0].a_type == A_FLOAT){ // TODO: float or int
            pyInletValue = PyLong_FromLong(av[0].a_w.w_float);
            PyTuple_SetItem(x->argsDict, 0, pyInletValue);
        }
        else if (av[0].a_type == A_SYMBOL){
            pyInletValue = PyUnicode_FromString(av[0].a_w.w_symbol->s_name);
            PyTuple_SetItem(x->argsDict, 0, pyInletValue);
        }
    }
    else{
        pyInletValue = PyList_New(ac + 1);
        PyList_SetItem(pyInletValue, 0, PyUnicode_FromString(s->s_name));
        for (int i = 0; i < ac; i++){
            if (av[i].a_type == A_FLOAT){ 
                int isInt = (int)av[i].a_w.w_float == av[i].a_w.w_float;
                if (isInt){
                    PyList_SetItem(pyInletValue, i + 1, PyLong_FromLong(av[i].a_w.w_float));
                }
                else{
                    PyList_SetItem(pyInletValue, i + 1, PyFloat_FromDouble(av[i].a_w.w_float));
                }
            }
            else if (av[i].a_type == A_SYMBOL){
                PyList_SetItem(pyInletValue, i + 1, PyUnicode_FromString(av[i].a_w.w_symbol->s_name));
            }
        }
        PyTuple_SetItem(x->argsDict, 0, pyInletValue);
    }

    // odd code, but solve the bug
    t_py *prev_obj;
    int prev_obj_exists = 0;
    PyObject *MainModule = PyModule_GetDict(PyImport_AddModule("__main__"));
    PyObject *oldObjectCapsule;
    if (MainModule != NULL) {
        oldObjectCapsule = PyDict_GetItemString(MainModule, "py4pd"); // borrowed reference
        if (oldObjectCapsule != NULL) {
            PyObject *pd_module = PyImport_ImportModule("__main__");
            PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, "py4pd");
            prev_obj = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
            prev_obj_exists = 1;
        }
    }

    PyObject *objectCapsule = py4pd_add_pd_object(x);
    if (objectCapsule == NULL){
        pd_error(x, "[Python] Failed to add object to Python");
        return;
    }
    pValue = PyObject_CallObject(x->function, x->argsDict);

    // odd code, but solve the bug
    if (prev_obj_exists == 1) {
        objectCapsule = py4pd_add_pd_object(prev_obj);
        if (objectCapsule == NULL){
            pd_error(x, "[Python] Failed to add object to Python");
            return;
        }
    }
    if (pValue != NULL) { 
        py4pd_convert_to_pd(x, pValue); 
    }
    else{
        Py_XDECREF(pValue);
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[Python] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        PyErr_Clear();
    }
    return;
}

// ======================== py_Object
void py_Object(t_py *x, t_atom *argv){
    // convert pointer to PyObject using pointer_to_pyobject
    PyObject *pValue;
    PyObject *pArg;
    pArg = pointer_to_pyobject(argv);
    if (pArg == NULL) {
        pd_error(x, "[py4pd] The pointer is not a PyObject!");
        return;
    }
    PyTuple_SetItem(x->argsDict, 0, pArg);
    t_py *prev_obj;
    int prev_obj_exists = 0;
    PyObject *MainModule = PyModule_GetDict(PyImport_AddModule("__main__"));
    PyObject *oldObjectCapsule;
    if (MainModule != NULL) {
        oldObjectCapsule = PyDict_GetItemString(MainModule, "py4pd"); // borrowed reference
        if (oldObjectCapsule != NULL) {
            PyObject *pd_module = PyImport_ImportModule("__main__");
            PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, "py4pd");
            prev_obj = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
            prev_obj_exists = 1;
        }
    }

    PyObject *objectCapsule = py4pd_add_pd_object(x);
    if (objectCapsule == NULL){
        pd_error(x, "[Python] Failed to add object to Python");
        return;
    }
    pValue = PyObject_CallObject(x->function, x->argsDict);

    // odd code, but solve the bug
    if (prev_obj_exists == 1) {
        objectCapsule = py4pd_add_pd_object(prev_obj);
        if (objectCapsule == NULL){
            pd_error(x, "[Python] Failed to add object to Python");
            return;
        }
    }
    if (pValue != NULL) { 
        py4pd_convert_to_pd(x, pValue); 
    }
    else{
        Py_XDECREF(pValue);
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[Python] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        PyErr_Clear();
    }
    return;
}

// =====================================
void *CreateNewObject(t_symbol *s, int argc, t_atom *argv) {
    (void) argc;
    (void) argv;
    const char *objectName = s->s_name;
    t_py *x = (t_py *)pd_new(pyNewObject);

    t_pd **py4pdInlet_proxies;

    x->visMode  = 0;
    x->x_canvas = canvas_getcurrent();       // pega o canvas atual
    t_canvas *c = x->x_canvas;               // get the current canvas
    t_symbol *patch_dir = canvas_getdir(c);  // directory of opened patch

    char py4pd_objectName[MAXPDSTRING];
    sprintf(py4pd_objectName, "py4pd_ObjectDict_%s", objectName);
    
    // ================================
    PyObject *pd_module = PyImport_ImportModule("__main__");
    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, py4pd_objectName);
    PyObject *PdDictCapsule = PyCapsule_GetPointer(py4pd_capsule, objectName);
    // ================================

    if (PdDictCapsule == NULL) {
        pd_error(x, "Error: PdDictCapsule is NULL");
        return NULL;
    }
    PyObject *PdDict = PyDict_GetItemString(PdDictCapsule, objectName);
    if (PdDict == NULL) {
        pd_error(x, "Error: PdDict is NULL");
        return NULL;
    }
    PyObject *pyFunction = PyDict_GetItemString(PdDict, "py4pdOBJFunction");
    if (pyFunction == NULL) {
        pd_error(x, "Error: pyFunction is NULL");
        return NULL;
    }
    
    PyObject *pyOUT = PyDict_GetItemString(PdDict, "py4pdOBJpyout");
    x->outPyPointer = PyLong_AsLong(pyOUT);
    PyObject *nooutlet = PyDict_GetItemString(PdDict, "py4pdOBJnooutlet");
    int nooutlet_int = PyLong_AsLong(nooutlet);
    x->function_called = 1;
    x->function = pyFunction;
    x->function_name = s;
    x->home_path = patch_dir;         // set name of the home path
    x->packages_path = patch_dir;     // set name of the packages path
    set_py4pd_config(x);  // set the config file (in py4pd.cfg, make this be
    py4pd_tempfolder(x);  // find the py4pd folder
    findpy4pd_folder(x);  // find the py4pd object folder
        
    // check if function use *args or **kwargs
    PyCodeObject* code = (PyCodeObject*)PyFunction_GetCode(pyFunction);
    if (code->co_flags & CO_VARARGS) {
        x->py_arg_numbers = 1;
        int argsNumberDefined = 0;
        int i;
        for (i = 0; i < argc; i++) {
            if (argv[i].a_type == A_SYMBOL) {
                if (strcmp(argv[i].a_w.w_symbol->s_name, "-n_args") == 0) {
                    if (i + 1 < argc) {
                        if (argv[i + 1].a_type == A_FLOAT) {
                            x->py_arg_numbers = (int)argv[i + 1].a_w.w_float;
                            argsNumberDefined = 1;
                        }
                        else {
                            pd_error(x, "[%s] function uses *args, you need to specify the number of arguments", objectName);
                            return NULL;
                        }
                    }
                    else {
                        pd_error(x, "[%s] this function uses *args, you need to specify the number of arguments using -n_args {number}", objectName);
                        return NULL;
                    }
                }
            }
        }
        if (argsNumberDefined == 0) {
            pd_error(x, "[%s] this function uses *args, you need to specify the number of arguments using -n_args {number}", objectName);
            return NULL;
        }
    }
    else if (code->co_flags & CO_VARKEYWORDS) {
        x->py_arg_numbers = 1;
        post("py4pd: function %s use **kwargs", objectName);
    }
    else {
        x->py_arg_numbers = code->co_argcount;
        post("py4pd: function %s use %d arguments", objectName, x->py_arg_numbers);
    }
    x->script_name = gensym(PyUnicode_AsUTF8(code->co_filename));

    int i;
    int pyFuncArgs = x->py_arg_numbers - 1;

    // Set inlet for all functions
    if (pyFuncArgs != 0){
        py4pdInlet_proxies = (t_pd **)getbytes((pyFuncArgs + 1) * sizeof(*py4pdInlet_proxies));
        for (i = 0; i < pyFuncArgs; i++){
                py4pdInlet_proxies[i] = pd_new(py4pdInlets_proxy_class);
                t_py4pdInlet_proxy *y = (t_py4pdInlet_proxy *)py4pdInlet_proxies[i];
                y->p_master = x;
                y->inletIndex = i + 1;
                inlet_new((t_object *)x, (t_pd *)y, 0, 0);
        }
        // set all args for Python Function to None, this prevent errors when users don't send all args
        int argNumbers = x->py_arg_numbers;
        x->argsDict = PyTuple_New(argNumbers);
        for (i = 0; i < argNumbers; i++) {
            PyTuple_SetItem(x->argsDict, i, Py_None);
        }
    }
    else{
        x->argsDict = PyTuple_New(1);
        PyTuple_SetItem(x->argsDict, 0, Py_None);
    }

    if (nooutlet_int == 0){
        x->out_A = outlet_new(&x->x_obj, 0);
    }
    return (x);
}

// =====================================
// =====================================
void *CreateNew_VISObject(t_symbol *s, int argc, t_atom *argv) {
    (void) argc;
    (void) argv;
    const char *objectName = s->s_name;
    t_py *x = (t_py *)pd_new(pyNewObject_VIS);
    t_pd **py4pdInlet_proxies;
    x->pyObject = 1;
    x->visMode  = 1;
    x->x_canvas = canvas_getcurrent();       // pega o canvas atual
    t_canvas *c = x->x_canvas;               // get the current canvas
    t_symbol *patch_dir = canvas_getdir(c);  // directory of opened patch

    char py4pd_objectName[MAXPDSTRING];
    sprintf(py4pd_objectName, "py4pd_ObjectDict_%s", objectName);
    
    // ================================
    PyObject *pd_module = PyImport_ImportModule("__main__");
    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, py4pd_objectName);
    PyObject *PdDictCapsule = PyCapsule_GetPointer(py4pd_capsule, objectName);
    // ================================
    if (PdDictCapsule == NULL) {
        pd_error(x, "Error: PdDictCapsule is NULL");
        return NULL;
    }

    PyObject *PdDict = PyDict_GetItemString(PdDictCapsule, objectName);
    if (PdDict == NULL) {
        pd_error(x, "Error: PdDict is NULL");
        return NULL;
    }

    PyObject *pyFunction = PyDict_GetItemString(PdDict, "py4pdOBJFunction");
    if (pyFunction == NULL) {
        pd_error(x, "Error: pyFunction is NULL");
        return NULL;
    }

    // ==================
    PyObject *pyOUT = PyDict_GetItemString(PdDict, "py4pdOBJpyout");
    x->outPyPointer = PyLong_AsLong(pyOUT);
    // Vis config
    t_symbol *py4pdArgs = gensym("-canvas");
    py4pd_InitVisMode(x, c, py4pdArgs, 0, argc, argv);
    x->function_called = 1;
    x->function = pyFunction;
    x->function_name = s;
    x->home_path = patch_dir;         // set name of the home path
    x->packages_path = patch_dir;     // set name of the packages path

    // get width and height of py4pdOBJwidth and py4pdOBJheight
    PyObject *py4pdOBJwidth = PyDict_GetItemString(PdDict, "py4pdOBJwidth");
    x->x_width = PyLong_AsLong(py4pdOBJwidth);
    PyObject *py4pdOBJheight = PyDict_GetItemString(PdDict, "py4pdOBJheight");
    x->x_height = PyLong_AsLong(py4pdOBJheight);

    set_py4pd_config(x);  // set the config file (in py4pd.cfg, make this be
    py4pd_tempfolder(x);  // find the py4pd folder
    findpy4pd_folder(x);  // find the py4pd object folder
    PyObject *inspect = NULL, *getfullargspec = NULL;
    PyObject *argspec = NULL, *argsFunc = NULL;
    inspect = PyImport_ImportModule("inspect");
    getfullargspec = PyObject_GetAttrString(inspect, "getfullargspec");
    argspec = PyObject_CallFunctionObjArgs(getfullargspec, pyFunction, NULL);
    argsFunc = PyTuple_GetItem(argspec, 0);       
    int py_args = PyObject_Size(argsFunc);
    x->py_arg_numbers = py_args;
    int i;
    int pyFuncArgs = x->py_arg_numbers - 1;

    // Set inlet for all functions
    if (pyFuncArgs != 0){
        py4pdInlet_proxies = (t_pd **)getbytes((pyFuncArgs + 1) * sizeof(*py4pdInlet_proxies));
        for (i = 0; i < pyFuncArgs; i++){
                py4pdInlet_proxies[i] = pd_new(py4pdInlets_proxy_class);
                t_py4pdInlet_proxy *y = (t_py4pdInlet_proxy *)py4pdInlet_proxies[i];
                y->p_master = x;
                y->inletIndex = i + 1;
                inlet_new((t_object *)x, (t_pd *)y, 0, 0);
        }
        // set all args for Python Function to None, this prevent errors when users don't send all args
        int argNumbers = x->py_arg_numbers;
        x->argsDict = PyTuple_New(argNumbers);
        for (i = 0; i < argNumbers; i++) {
            PyTuple_SetItem(x->argsDict, i, Py_None);
        }
    }
    else{
        x->argsDict = PyTuple_New(1);
        PyTuple_SetItem(x->argsDict, 0, Py_None);
    }
    x->out_A = outlet_new(&x->x_obj, 0);
    return (x);
}

// =====================================
void *pyObjectFree(t_py *x) {
    if (x->visMode != 0) {
        PY4PD_free(x);
    }
    return (void *)x;

}

// =====================================
/**
 * @brief add new Python Object to PureData
 * @param x
 * @param argc
 * @param argv
 * @return
 */
PyObject *pdAddPyObject(PyObject *self, PyObject *args, PyObject *keywords) {
    (void)self;
    char *objectName;
    PyObject *Function;
    int w = 250, h = 250;
    int objpyout = 0;
    int nooutlet = 0;
    int added2pd_info = 0;
    if (!PyArg_ParseTuple(args, "Os", &Function, &objectName)) { 
        post("[Python]: Error parsing arguments");
        return NULL;
    }
    const char *objectType = "NORMAL";
    if (keywords != NULL) {
        if (PyDict_Contains(keywords, PyUnicode_FromString("objtype"))) {
            PyObject *type = PyDict_GetItemString(keywords, "objtype");
            objectType = PyUnicode_AsUTF8(type);
        }
        if(PyDict_Contains(keywords, PyUnicode_FromString("figsize"))) {
            PyObject *figsize = PyDict_GetItemString(keywords, "figsize"); // this is defined in python using figsize=(w,h)
            PyObject *width = PyTuple_GetItem(figsize, 0);
            PyObject *height = PyTuple_GetItem(figsize, 1);
            w = PyLong_AsLong(width);
            h = PyLong_AsLong(height);
        }
        if (PyDict_Contains(keywords, PyUnicode_FromString("pyout"))) {
            PyObject *output = PyDict_GetItemString(keywords, "pyout"); // it gets the data type output
            if (output == Py_True) {
                objpyout = 1;
            }
        }
        if (PyDict_Contains(keywords, PyUnicode_FromString("no_outlet"))) {
            PyObject *output = PyDict_GetItemString(keywords, "no_outlet"); // it gets the data type output
            if (output == Py_True) {
                nooutlet = 1;
            }
        }
        if (PyDict_Contains(keywords, PyUnicode_FromString("added2pd_info"))) {
            PyObject *output = PyDict_GetItemString(keywords, "added2pd_info"); // it gets the data type output
            if (output == Py_True) {
                added2pd_info = 1;
            }
        }
    }

    // add object to main dict
    PyObject *nestedDict = PyDict_New();
    PyDict_SetItemString(nestedDict, "py4pdOBJFunction", Function);
    PyDict_SetItemString(nestedDict, "py4pdOBJwidth", PyLong_FromLong(w));
    PyDict_SetItemString(nestedDict, "py4pdOBJheight", PyLong_FromLong(h));
    PyDict_SetItemString(nestedDict, "py4pdOBJpyout", PyLong_FromLong(objpyout));
    PyDict_SetItemString(nestedDict, "py4pdOBJnooutlet", PyLong_FromLong(nooutlet));
    PyObject *objectDict = PyDict_New();
    PyDict_SetItemString(objectDict, objectName, nestedDict);
    PyObject *py4pd_capsule = PyCapsule_New(objectDict, objectName, NULL);
    char py4pd_objectName[MAXPDSTRING];
    sprintf(py4pd_objectName, "py4pd_ObjectDict_%s", objectName);
    PyModule_AddObject(PyImport_ImportModule("__main__"), py4pd_objectName, py4pd_capsule);

    
    // get number of args, BUG: in build-in functions, this will return 0
    PyObject *inspect = NULL, *getfullargspec = NULL;
    PyObject *argspec = NULL, *argsFunc = NULL;
    inspect = PyImport_ImportModule("inspect");
    getfullargspec = PyObject_GetAttrString(inspect, "getfullargspec");
    argspec = PyObject_CallFunctionObjArgs(getfullargspec, Function, NULL); // it returns -1 for built-in functions
    if (argspec == NULL) {
        // get python function name
        PyObject *pyFunctionName = PyObject_GetAttrString(Function, "__name__");
        post("[Python]: Error getting the number of arguments for function {%s}", PyUnicode_AsUTF8(pyFunctionName));
        Py_DECREF(pyFunctionName);
        return NULL;
    }
    argsFunc = PyTuple_GetItem(argspec, 0);       
    int py_args = PyObject_Size(argsFunc);

    // NORMAL
    if ((strcmp(objectType, "NORMAL") == 0)){
        pyNewObject = class_new(gensym(objectName), (t_newmethod)CreateNewObject, 0, sizeof(t_py), CLASS_DEFAULT, A_GIMME, 0);
        class_addmethod(pyNewObject, (t_method)py_Object, gensym("PyObject"), A_POINTER, 0);
        class_addmethod(pyNewObject, (t_method)documentation, gensym("doc"), 0, 0);
        class_addmethod(pyNewObject, (t_method)usepointers, gensym("pointers"), A_FLOAT, 0);
        class_addanything(pyNewObject, py_anything);

    }
    // VIS
    else if ((strcmp(objectType, "VIS") == 0)){
        pyNewObject_VIS = class_new(gensym(objectName), (t_newmethod)CreateNew_VISObject, (t_method)pyObjectFree, sizeof(t_py), CLASS_DEFAULT, A_GIMME, 0);
        class_addanything(pyNewObject_VIS, py_anything);
        class_addmethod(pyNewObject_VIS, (t_method)PY4PD_zoom, gensym("zoom"), A_CANT, 0);
        class_addmethod(pyNewObject, (t_method)py_Object, gensym("PyObject"), A_POINTER, 0);
        class_addmethod(pyNewObject, (t_method)documentation, gensym("doc"), 0, 0);
    }
    // AUDIOIN
    else if ((strcmp(objectType, "AUDIOIN") == 0)){
        pyNewObject_AudioIn = NULL;
        class_addmethod(pyNewObject, (t_method)documentation, gensym("doc"), 0, 0);

    }
    // AUDIOOUT
    else if ((strcmp(objectType, "AUDIOOUT") == 0)){
        pyNewObject_AudioOut = NULL;
        class_addmethod(pyNewObject, (t_method)documentation, gensym("doc"), 0, 0);

    }
    // AUDIO
    else if ((strcmp(objectType, "AUDIO") == 0)){
        pyNewObject_Audio = NULL;
        class_addmethod(pyNewObject, (t_method)documentation, gensym("doc"), 0, 0);
    }
    
    if (py_args != 0){
        py4pdInlets_proxy_class = class_new(gensym("_py4pdInlets_proxy"), 0, 0, sizeof(t_py4pdInlet_proxy), CLASS_DEFAULT, 0);
        class_addanything(py4pdInlets_proxy_class, py4pdInlets_proxy_anything);
        class_addlist(py4pdInlets_proxy_class, py4pdInlets_proxy_list);
        class_addmethod(py4pdInlets_proxy_class, (t_method)py4pdInlets_proxy_pointer, gensym("PyObject"), A_POINTER, 0);
    }
    if (added2pd_info == 1){
        post("[py4pd]: Object {%s} added to PureData", objectName);
    }


    return PyLong_FromLong(1);

}
