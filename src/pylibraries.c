#include "pylibraries.h"
#include "py4pd.h"
#include "py4pd_utils.h"
#include "py4pd_pic.h"

static t_class *pyNewObject;
static t_class *pyNewObject_VIS;
static t_class *pyNewObject_AudioIn;
static t_class *pyNewObject_AudioOut;

static t_class *py4pdInlets_proxy_class;

// =====================================
void py4pdInlets_proxy_anything(t_py4pdInlet_proxy *x, t_symbol *s, int ac, t_atom *av){
    t_py *py4pd = (t_py *)x->p_master;
    if (ac == 0){
        // convert *s to string in Python
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

    // CONVERT TO PYTHON OBJECTS
    if (ac == 0){
        pyInletValue = PyUnicode_FromString(s->s_name);
        PyTuple_SetItem(py4pd->argsDict, 0, pyInletValue);
    }
    else if (ac == 1){
        if (av[0].a_type == A_FLOAT){
            pyInletValue = PyLong_FromLong(av[0].a_w.w_float);
            PyTuple_SetItem(py4pd->argsDict, x->inletIndex, pyInletValue);
        }
        else if (av[0].a_type == A_SYMBOL){
            pyInletValue = PyUnicode_FromString(av[0].a_w.w_symbol->s_name);
            PyTuple_SetItem(py4pd->argsDict, x->inletIndex, pyInletValue);
        }
    }
    else { // NOTE: Need some work here
        pyInletValue = PyList_New(ac);
        for (int i = 0; i < ac; i++){
            if (av[i].a_type == A_FLOAT){ // TODO: check if it is an int or a float
                PyList_SetItem(pyInletValue, i, PyLong_FromLong(av[i].a_w.w_float));
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
void py_anything(t_py *x, t_symbol *s, int ac, t_atom *av){
    
    PyObject *pyInletValue;
    
    // CONVERT TO PYTHON OBJECTS
    if (ac == 0){
        pyInletValue = PyUnicode_FromString(s->s_name);
        PyTuple_SetItem(x->argsDict, 0, pyInletValue);
    }
    else if ((s == gensym("list") || s == gensym("anything")) && ac > 0){
        pyInletValue = PyList_New(ac);
        for (int i = 0; i < ac; i++){
            if (av[i].a_type == A_FLOAT){ // TODO: check if it is an int or a float
                PyList_SetItem(pyInletValue, i, PyLong_FromLong(av[i].a_w.w_float));
            }
            else if (av[i].a_type == A_SYMBOL){
                PyList_SetItem(pyInletValue, i, PyUnicode_FromString(av[i].a_w.w_symbol->s_name));
            }
        }
        PyTuple_SetItem(x->argsDict, 0, pyInletValue);
    }
    else if ((s == gensym("float") || s == gensym("symbol")) && ac == 1){
        if (av[0].a_type == A_FLOAT){
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
            if (av[i].a_type == A_FLOAT){ // TODO: check if it is an int or a float
                PyList_SetItem(pyInletValue, i + 1, PyLong_FromLong(av[i].a_w.w_float));
            }
            else if (av[i].a_type == A_SYMBOL){
                PyList_SetItem(pyInletValue, i + 1, PyUnicode_FromString(av[i].a_w.w_symbol->s_name));
            }
        }
        PyTuple_SetItem(x->argsDict, 0, pyInletValue);
    }
    // see what is inside x->argsDict

    // RUN THE FUNCTION
    PyObject *objectCapsule = py4pd_add_pd_object(x);
    if (objectCapsule == NULL){
        pd_error(x, "[Python] Failed to add object to Python");
        return;
    }
    PyObject *pValue = PyObject_CallObject(x->function, x->argsDict);

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
void *py_newObject(t_symbol *s, int argc, t_atom *argv) {
    (void) argc;
    (void) argv;
    const char *objectName = s->s_name;
    t_py *x = (t_py *)pd_new(pyNewObject);

    t_pd **py4pdInlet_proxies;

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

    PyObject *moduleName = PyDict_GetItemString(PdDict, "moduleName");
    if (moduleName == NULL) {
        pd_error(x, "Error: moduleName is NULL");
        return NULL;
    }

    PyObject *functionName = PyDict_GetItemString(PdDict, "functionName");
    if (functionName == NULL) {
        pd_error(x, "Error: functionName is NULL");
        return NULL;
    }

    PyObject *objectType = PyDict_GetItemString(PdDict, "objectType");
    if (objectType == NULL) {
        pd_error(x, "Error: objectType is NULL");
        return NULL;
    }

    x->home_path = patch_dir;         // set name of the home path
    x->packages_path = patch_dir;     // set name of the packages path
    set_py4pd_config(x);  // set the config file (in py4pd.cfg, make this be
    py4pd_tempfolder(x);  // find the py4pd folder
    findpy4pd_folder(x);  // find the py4pd object folder

    t_atom setArgs[2];
    setArgs[0].a_type = A_SYMBOL;
    setArgs[0].a_w.w_symbol = gensym(PyUnicode_AsUTF8(moduleName));
    setArgs[1].a_type = A_SYMBOL;
    setArgs[1].a_w.w_symbol = gensym(PyUnicode_AsUTF8(functionName));
    x->py4pd_lib = 1;

    set_function(x, NULL, 2, setArgs);
    int i;
    int pyFuncArgs = x->py_arg_numbers - 1;

    py4pdInlet_proxies = (t_pd **)getbytes((pyFuncArgs + 1) * sizeof(*py4pdInlet_proxies));
    for (i = 0; i < pyFuncArgs; i++){

        // char methodForInlet[MAXPDSTRING];
        // if (i < 9) {
            // sprintf(methodForInlet, "addInlet 0%d", i + 1);
        // }
        // else {
            // sprintf(methodForInlet, "addInlet %d", i + 1);
        // }
            py4pdInlet_proxies[i] = pd_new(py4pdInlets_proxy_class);
        	t_py4pdInlet_proxy *y = (t_py4pdInlet_proxy *)py4pdInlet_proxies[i];
            y->p_master = x;
            y->inletIndex = i + 1;
            inlet_new((t_object *)x, (t_pd *)y, 0, 0);
    }

    x->out_A = outlet_new(&x->x_obj, 0);
    int argNumbers = x->py_arg_numbers;
    x->argsDict = PyTuple_New(argNumbers);
    for (i = 0; i < argNumbers; i++) {
        PyTuple_SetItem(x->argsDict, i, Py_None);
    }



    return (x);
}

// =====================================
/**
 * @brief add new Python Object to PureData
 * @param x
 * @param argc
 * @param argv
 * @return
 */

PyObject *pdAddPyObject(PyObject *self, PyObject *args) {
    (void)self;

    char *objectName;
    char *objectType;
    char *objectModule;
    char *objectFunction;

    // ================================
    PyObject *pd_module = PyImport_ImportModule("__main__");
    PyObject *py4pd_OBJECT = PyObject_GetAttrString(pd_module, "py4pd");
    t_py *py4pd = (t_py *)PyCapsule_GetPointer(py4pd_OBJECT, "py4pd");
    // ================================

    if (!PyArg_ParseTuple(args, "ssss", &objectName, &objectType, &objectModule, &objectFunction)) {
        post("[Python]: Error parsing arguments");
        return NULL;
    }

    PyObject *nestedDict = PyDict_New();
    PyDict_SetItemString(nestedDict, "functionName", PyUnicode_FromString(objectFunction));
    PyDict_SetItemString(nestedDict, "moduleName", PyUnicode_FromString(objectModule));
    PyDict_SetItemString(nestedDict, "objectType", PyUnicode_FromString(objectType));
    PyObject *objectDict = PyDict_New();
    PyDict_SetItemString(objectDict, objectName, nestedDict);
    PyObject *py4pd_capsule = PyCapsule_New(objectDict, objectName, NULL);
    char py4pd_objectName[MAXPDSTRING];
    sprintf(py4pd_objectName, "py4pd_ObjectDict_%s", objectName);
    PyModule_AddObject(PyImport_ImportModule("__main__"), py4pd_objectName, py4pd_capsule);

    // NORMAL
    if ((strcmp(objectType, "NORMAL") == 0)){
        pyNewObject = class_new(gensym(objectName), (t_newmethod)py_newObject, 0, sizeof(t_py), CLASS_DEFAULT, A_GIMME, 0);
        class_addanything(pyNewObject, py_anything);
        class_addmethod(pyNewObject, (t_method)reload, gensym("reload"), 0, 0);  // run python script
    }
    // VIS
    else if ((strcmp(objectType, "VIS") == 0)){
        pyNewObject_VIS = class_new(gensym(objectName), (t_newmethod)py_newObject, 0, sizeof(t_py), CLASS_DEFAULT, A_GIMME, 0);
        class_addanything(pyNewObject_VIS, py_anything);
        class_addmethod(pyNewObject_VIS, (t_method)reload, gensym("reload"), 0, 0);  // run python script
    }
    // AUDIOIN
    else if ((strcmp(objectType, "AUDIOIN") == 0)){

    }
    // AUDIOOUT
    else if ((strcmp(objectType, "AUDIOOUT") == 0)){

    }
    // AUDIO
    else if ((strcmp(objectType, "AUDIO") == 0)){

    }

    py4pdInlets_proxy_class = class_new(gensym("_py4pdInlets_proxy"), 0, 0, sizeof(t_py4pdInlet_proxy), CLASS_DEFAULT, 0);
    class_addanything(py4pdInlets_proxy_class, py4pdInlets_proxy_anything);
    class_addlist(py4pdInlets_proxy_class, py4pdInlets_proxy_list);

    return PyLong_FromLong(1);
}
