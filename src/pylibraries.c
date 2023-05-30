#include "pylibraries.h"
#include "m_pd.h"
#include "py4pd_utils.h"
#include "py4pd_pic.h"
#include "py4pd.h"

#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#include <numpy/arrayobject.h>

t_class *pyNewObject_VIS;

static t_class *pyNewObject;
static t_class *pyNewObject_AudioIn;
static t_class *pyNewObject_AudioOut;
static t_class *pyNewObject_Audio;

static t_class *py4pdInlets_proxy_class;

// ====================================================
void create_pyObject_inlets(t_py *x, int argc, t_atom *argv) {
    t_pd **py4pdInlet_proxies;
    int i;
    int pyFuncArgs = x->py_arg_numbers - 1;
    if (pyFuncArgs != 0){
        py4pdInlet_proxies = (t_pd **)getbytes((pyFuncArgs + 1) * sizeof(*py4pdInlet_proxies));

        // ===========================
        // It creates the inlet proxies
        for (i = 0; i < pyFuncArgs; i++){
                py4pdInlet_proxies[i] = pd_new(py4pdInlets_proxy_class);
                t_py4pdInlet_proxy *y = (t_py4pdInlet_proxy *)py4pdInlet_proxies[i];
                y->p_master = x;
                y->inletIndex = i + 1;
                inlet_new((t_object *)x, (t_pd *)y, 0, 0);
        }
        // ===========================
        int argNumbers = x->py_arg_numbers;
        x->argsDict = PyTuple_New(argNumbers);
        for (i = 0; i < argNumbers; i++) {
            if (i <= argc) {
                if (argv[i].a_type == A_FLOAT) {
                    char pd_atom[64];
                    atom_string(&argv[i], pd_atom, 64);
                    if (strchr(pd_atom, '.') != NULL) {
                        PyTuple_SetItem(x->argsDict, i, PyFloat_FromDouble(argv[i].a_w.w_float));
                    }
                    else{
                        PyTuple_SetItem(x->argsDict, i, PyLong_FromLong(argv[i].a_w.w_float));
                    }
                }

                else if (argv[i].a_type == A_SYMBOL) {
                    if (strcmp(argv[i].a_w.w_symbol->s_name, "None") == 0) {
                        PyTuple_SetItem(x->argsDict, i, Py_None);
                    }
                    else {
                        PyTuple_SetItem(x->argsDict, i, PyUnicode_FromString(argv[i].a_w.w_symbol->s_name));
                    }
                }

                else {
                    PyTuple_SetItem(x->argsDict, i, Py_None);
                }
            }
            else{
                PyTuple_SetItem(x->argsDict, i, Py_None);
            } 
        }
    }
    else{
        x->argsDict = PyTuple_New(1);
        PyTuple_SetItem(x->argsDict, 0, Py_None);
    }
}

// =====================================
void py4pdObjPic_save(t_gobj *z, t_binbuf *b){ 
    t_py *x = (t_py *)z;
    binbuf_addv(b, "ssii", gensym("#X"), gensym("obj"), x->x_obj.te_xpix, x->x_obj.te_ypix);
    binbuf_addbinbuf(b, ((t_py *)x)->x_obj.te_binbuf);

    int objAtomsCount = binbuf_getnatom(((t_py *)x)->x_obj.te_binbuf);
    if (objAtomsCount == 1){
        binbuf_addv(b, "ii", x->x_width, x->x_height);
    }
    binbuf_addsemi(b);
    return;
}

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
        PyObject *pyInletValue = PyList_New(ac + 1);
        PyList_SetItem(pyInletValue, 0, PyUnicode_FromString(s->s_name));
        for (int i = 0; i < ac; i++){
            if (av[i].a_type == A_FLOAT){ // TODO: check if it is an int or a float
                int isInt = (int)av[0].a_w.w_float == av[0].a_w.w_float;
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
        PyTuple_SetItem(py4pd->argsDict, x->inletIndex, pyInletValue);
    }
    return;
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
    // TODO: revisar, error quando pValue is null and we add a new object
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

    if (x->audioOutput == 1){
        return;
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
    if (prev_obj_exists == 1 && pValue != NULL) {
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
        pd_error(x, "[%s] Call failed: %s", x->objectName->s_name, PyUnicode_AsUTF8(pstr));
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

    // If there is a previous object and a value is passed, 
    // create a Python object capsule and add the previous object to it, 
    // so that when the current object is connected to a Python object, 
    // its output is sent to the right outlet.
    if (prev_obj_exists == 1 && pValue != NULL) {
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
void reloadObject(t_py *x){
    // reload script_name of *x (script_name is a t_symbol for path of script)
    const char* full_path = x->script_name->s_name;
    const char* separator = strrchr(full_path, '/');
    if (separator == NULL) {
        separator = strrchr(full_path, '\\');
    }

    if (separator != NULL) {
        const char* tree = separator + 1;
        const char* extension = strstr(tree, ".py");
        if (extension != NULL) {
            size_t len = extension - tree;
            char* name = (char*)malloc(len + 1);
            memcpy(name, tree, len);
            name[len] = '\0';
            PyObject *pName = PyUnicode_FromString(name);
            PyObject *pModule = PyImport_Import(pName);
            // reload module
            PyObject *pModuleReloaded = PyImport_ReloadModule(pModule);
            if (pModuleReloaded == NULL) {
                pd_error(x, "[Python] Failed to reload module");
            }
            x->function = PyObject_GetAttrString(pModuleReloaded, x->function_name->s_name);
            if (x->function == NULL) {
                pd_error(x, "[Python] Failed to get function");
                free(name);
                return;
            }
            else {
                post("[Python] Function reloaded");
            }
            free(name);
        }
        else {
            pd_error(x, "[Python] Invalid script name");
        }
    }
}

// =====================================
t_int *library_AudioIN_perform(t_int *w) {
    t_py *x = (t_py *)(w[1]);  // this is the object itself
    t_sample *audioIn = (t_sample *)(w[2]);  // this is the input vector (the sound)
    int n = (int)(w[3]);

    PyObject *pValue, *pAudio; 
    const npy_intp dims = n;
    pAudio = PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT, audioIn);
    PyTuple_SetItem(x->argsDict, 0, pAudio);

    PyObject *objectCapsule = py4pd_add_pd_object(x);
    if (objectCapsule == NULL){
        pd_error(x, "[Python] Failed to add object to Python");
        return (w + 4);
    }
    pValue = PyObject_CallObject(x->function, x->argsDict);

    if (pValue != NULL) {
        py4pd_convert_to_pd(x, pValue);  // convert the value to pd
    } 
    else {                             // if the function returns a error
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        PyErr_Clear();
    }
    Py_XDECREF(pValue);
    return (w + 4);
}

// =====================================
t_int *library_AudioOUT_perform(t_int *w) {
    t_py *x = (t_py *)(w[1]);  // this is the object itself
    t_sample *audioOut = (t_sample *)(w[2]);
    int n = (int)(w[3]);
    PyObject *pValue; 

    // TODO: add old capsule

    PyObject *objectCapsule = py4pd_add_pd_object(x);
    if (objectCapsule == NULL){
        pd_error(x, "[Python] Failed to add object to Python");
        return (w + 4);
    }
    pValue = PyObject_CallObject(x->function, x->argsDict);

    if (pValue != NULL) {
        if (PyArray_Check(pValue)) {
            PyArrayObject *pArray = PyArray_GETCONTIGUOUS((PyArrayObject *)pValue);
            PyArray_Descr *pArrayType = PyArray_DESCR(pArray);
            int arrayLength = PyArray_SIZE(pArray);
            if (arrayLength == n && PyArray_NDIM(pArray) == 1) {
                if (pArrayType->type_num == NPY_INT) {
                    pd_error(x, "[py4pd] The numpy array must be float, returned int");
                }
                else if (pArrayType->type_num == NPY_FLOAT) {
                    float *audioFloat = (float*)PyArray_DATA(pValue);
                    for (int i = 0; i < n; i++) {
                        audioOut[i] = (t_sample)audioFloat[i];
                    }
                }
                else if (pArrayType->type_num == NPY_DOUBLE) {
                    double *audioDouble = (double*)PyArray_DATA(pValue);
                    for (int i = 0; i < n; i++) {
                        audioOut[i] = (t_sample)audioDouble[i];
                    }
                }
                else {
                    pd_error(x, "[py4pd] The numpy array must be float, returned %d", pArrayType->type_num);
                }
                Py_DECREF(pArrayType);
                Py_DECREF(pArray);
            }
            else {
                pd_error(x, "[py4pd] The numpy array must have the same length of the vecsize and 1 dim. Returned: %d samples and %d dims", arrayLength, PyArray_NDIM(pArray));
                Py_DECREF(pArrayType);
                Py_DECREF(pArray);
            }
        } 
        else{
            pd_error(x, "[Python] Python function must return a numpy array");
        }
    } 
    else {                             // if the function returns a error
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        PyErr_Clear();
    }
    Py_XDECREF(pValue);
    return (w + 4);
}

// =====================================
t_int *library_Audio_perform(t_int *w) {
    t_py *x = (t_py *)(w[1]);  // this is the object itself
    t_sample *audioIn = (t_sample *)(w[2]);  // this is the input vector (the sound)
    t_sample *audioOut = (t_sample *)(w[3]);
    int n = (int)(w[4]);
    PyObject *pValue, *pAudio; 

    const npy_intp dims = n;
    pAudio = PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT, audioIn);
    PyTuple_SetItem(x->argsDict, 0, pAudio);

    PyObject *objectCapsule = py4pd_add_pd_object(x);
    if (objectCapsule == NULL){
        pd_error(x, "[Python] Failed to add object to Python");
        return (w + 5);
    }
    pValue = PyObject_CallObject(x->function, x->argsDict);
    if (pValue != NULL) {
        if (PyArray_Check(pValue)) {
            PyArrayObject *pArray = PyArray_GETCONTIGUOUS((PyArrayObject *)pValue);
            PyArray_Descr *pArrayType = PyArray_DESCR(pArray);
            int arrayLength = PyArray_SIZE(pArray);
            if (arrayLength == n && PyArray_NDIM(pArray) == 1) {
                if (pArrayType->type_num == NPY_INT) {
                    pd_error(x, "[py4pd] The numpy array must be float, returned int");
                }
                else if (pArrayType->type_num == NPY_FLOAT) {
                    float *audioFloat = (float*)PyArray_DATA(pValue);
                    for (int i = 0; i < n; i++) {
                        audioOut[i] = (t_sample)audioFloat[i];
                    }
                }
                else if (pArrayType->type_num == NPY_DOUBLE) {
                    double *audioDouble = (double*)PyArray_DATA(pValue);
                    for (int i = 0; i < n; i++) {
                        audioOut[i] = (t_sample)audioDouble[i];
                    }
                }
                else {
                    pd_error(x, "[py4pd] The numpy array must be float, returned %d", pArrayType->type_num);
                }
                Py_DECREF(pArrayType);
                Py_DECREF(pArray);
            }
            else {
                pd_error(x, "[py4pd] The numpy array must have the same length of the vecsize and 1 dim. Returned: %d samples and %d dims", arrayLength, PyArray_NDIM(pArray));
                Py_DECREF(pArrayType);
                Py_DECREF(pArray);
            }
        } 
        else{
            pd_error(x, "[Python] Python function must return a numpy array");
        }
    } 
    else {                             // if the function returns a error
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        PyErr_Clear();
    }
    Py_XDECREF(pValue);
    return (w + 5);
}

// =====================================
static void library_dsp(t_py *x, t_signal **sp) {
    if (x->audioOutput == 0) {
        dsp_add(library_AudioIN_perform, 3, x, sp[0]->s_vec, sp[0]->s_n);
    } 
    else if ((x->audioInput == 0) && (x->audioOutput == 1)){
        dsp_add(library_AudioOUT_perform, 3, x, sp[0]->s_vec, sp[0]->s_n);
    }
    else if ((x->audioInput == 1) && (x->audioOutput == 1)) {  // python output is audio
        dsp_add(library_Audio_perform, 4, x, sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_n);
    }
}

// =====================================
void *New_NORMAL_Object(t_symbol *s, int argc, t_atom *argv) {
    const char *objectName = s->s_name;
    t_py *x = (t_py *)pd_new(pyNewObject);
    x->visMode  = 0;
    x->pyObject = 1;
    x->x_canvas = canvas_getcurrent();       // pega o canvas atual
    t_canvas *c = x->x_canvas;               // get the current canvas
    t_symbol *patch_dir = canvas_getdir(c);  // directory of opened patch
    char py4pd_objectName[MAXPDSTRING];
    sprintf(py4pd_objectName, "py4pd_ObjectDict_%s", objectName);
    x->objectName = gensym(objectName);
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
    PyObject *nooutlet = PyDict_GetItemString(PdDict, "py4pdOBJnooutlet");
    int nooutlet_int = PyLong_AsLong(nooutlet);
    x->outPyPointer = PyLong_AsLong(pyOUT);
    x->function_called = 1;
    x->function = pyFunction;
    x->pdPatchFolder = patch_dir;         // set name of the home path
    x->pkgPath = patch_dir;     // set name of the packages path
    x->py_arg_numbers = 0;
    set_py4pd_config(x);  // set the config file  TODO: I want to rethink this)
    py4pd_tempfolder(x);  // find the py4pd temp folder
    findpy4pd_folder(x);  // find the py4pd object folder
        
    // Parse args for translation between Pd and Python
    PyCodeObject *code = (PyCodeObject*)PyFunction_GetCode(pyFunction);
    x->function_name = gensym(PyUnicode_AsUTF8(code->co_name));
    x->script_name = gensym(PyUnicode_AsUTF8(code->co_filename));
    int parseArgsRight = libraries_parser_args(x, code, argc, argv); // NOTE: added
    if (parseArgsRight == 0) {
        return NULL;
    }
    create_pyObject_inlets(x, argc, argv);
    if (nooutlet_int == 0){
        x->out1 = outlet_new(&x->x_obj, 0);
    }
    object_count++;
    return (x);
}

// =====================================
void *New_VIS_Object(t_symbol *s, int argc, t_atom *argv) {
    (void) argc;
    (void) argv;
    const char *objectName = s->s_name;
    t_py *x = (t_py *)pd_new(pyNewObject_VIS);
    // t_pd **py4pdInlet_proxies;
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


    /* NOTE: For version 0.8.0
    PyObject *pyShowFunction = PyDict_GetItemString(PdDict, "py4pdOBJshowFunction");
    if (pyShowFunction == NULL){
        x->showFunction = NULL;
        logpost(x, 4, "showFunction was not defined");
    }
    else{
        x->showFunction = pyShowFunction;
        logpost(x, 4, "showFunction was defined");
    }
    */


    // ==================
    // PIC PIC
    PyObject *pyOUT = PyDict_GetItemString(PdDict, "py4pdOBJpyout");
    x->outPyPointer = PyLong_AsLong(pyOUT);
    t_symbol *py4pdArgs = gensym("-canvas");
    PyObject *py4pdOBJwidth = PyDict_GetItemString(PdDict, "py4pdOBJwidth");
    x->x_width = PyLong_AsLong(py4pdOBJwidth);
    PyObject *py4pdOBJheight = PyDict_GetItemString(PdDict, "py4pdOBJheight");
    x->x_height = PyLong_AsLong(py4pdOBJheight);
    if (argc > 1) {
        if (argv[0].a_type == A_FLOAT) {
            x->x_width = argv[0].a_w.w_float;
        }
        if (argv[1].a_type == A_FLOAT) {
            x->x_height = argv[1].a_w.w_float;
        }
    }
    py4pd_InitVisMode(x, c, py4pdArgs, 0, argc, argv);
    // PIC =============

    x->outPyPointer = PyLong_AsLong(pyOUT);
    PyObject *nooutlet = PyDict_GetItemString(PdDict, "py4pdOBJnooutlet");
    int nooutlet_int = PyLong_AsLong(nooutlet);

    // Ordinary Function
    x->function = pyFunction;
    x->function_called = 1;

    x->pdPatchFolder = patch_dir;         // set name of the home path
    x->pkgPath = patch_dir;     // set name of the packages path
    x->py_arg_numbers = 0;
    set_py4pd_config(x);  // set the config file (in py4pd.cfg, make this be
    py4pd_tempfolder(x);  // find the py4pd folder
    findpy4pd_folder(x);  // find the py4pd object folder
    // check if function use *args or **kwargs
    PyCodeObject* code = (PyCodeObject*)PyFunction_GetCode(pyFunction);
    x->function_name = gensym(PyUnicode_AsUTF8(code->co_name));
    x->script_name = gensym(PyUnicode_AsUTF8(code->co_filename));
    int parseArgsRight = libraries_parser_args(x, code, argc, argv); // NOTE: added
    if (parseArgsRight == 0) {
        return NULL;
    }
    create_pyObject_inlets(x, argc, argv);
    if (nooutlet_int == 0){
        x->out1 = outlet_new(&x->x_obj, 0);
    }
    object_count++;
    return (x);
}


// =====================================
void *New_AudioIN_Object(t_symbol *s, int argc, t_atom *argv) {
    const char *objectName = s->s_name;
    t_py *x = (t_py *)pd_new(pyNewObject_AudioIn);
    x->visMode  = 0;
    x->pyObject = 1;
    x->x_canvas = canvas_getcurrent();       // pega o canvas atual
    t_canvas *c = x->x_canvas;               // get the current canvas
    t_symbol *patch_dir = canvas_getdir(c);  // directory of opened patch
    char py4pd_objectName[MAXPDSTRING];
    sprintf(py4pd_objectName, "py4pd_ObjectDict_%s", objectName);
    x->objectName = gensym(objectName);
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
    PyObject *nooutlet = PyDict_GetItemString(PdDict, "py4pdOBJnooutlet");
    int nooutlet_int = PyLong_AsLong(nooutlet);
    x->outPyPointer = PyLong_AsLong(pyOUT);
    x->function_called = 1;
    x->function = pyFunction;
    x->pdPatchFolder = patch_dir;         // set name of the home path
    x->pkgPath = patch_dir;     // set name of the packages path
    x->py_arg_numbers = 0;
    set_py4pd_config(x);  // set the config file  TODO: I want to rethink this)
    py4pd_tempfolder(x);  // find the py4pd temp folder
    findpy4pd_folder(x);  // find the py4pd object folder
    // Parse args for translation between Pd and Python
    PyCodeObject *code = (PyCodeObject*)PyFunction_GetCode(pyFunction);
    x->function_name = gensym(PyUnicode_AsUTF8(code->co_name));
    x->script_name = gensym(PyUnicode_AsUTF8(code->co_filename));
    int parseArgsRight = libraries_parser_args(x, code, argc, argv); // NOTE: added
    if (parseArgsRight == 0) {
        return NULL;
    }
    create_pyObject_inlets(x, argc, argv);
    if (nooutlet_int == 0){
        x->out1 = outlet_new(&x->x_obj, 0);
    }
    // check if numpy array is imported
    int numpyArrayImported = _import_array();
    if (numpyArrayImported == 0) {
        x->numpyImported = 1;
    }
    else{
        x->numpyImported = 0;
        pd_error(x, "[py4pd] Not possible to import numpy array");
        return NULL;
    }
    object_count++;
    return (x);
}

// =====================================
void *New_AudioOUT_Object(t_symbol *s, int argc, t_atom *argv) {
    const char *objectName = s->s_name;
    t_py *x = (t_py *)pd_new(pyNewObject_AudioOut);
    x->visMode  = 0;
    x->pyObject = 1;
    x->audioInput = 0;
    x->audioOutput = 1;
    x->x_canvas = canvas_getcurrent();       // pega o canvas atual
    t_canvas *c = x->x_canvas;               // get the current canvas
    t_symbol *patch_dir = canvas_getdir(c);  // directory of opened patch
    char py4pd_objectName[MAXPDSTRING];
    sprintf(py4pd_objectName, "py4pd_ObjectDict_%s", objectName);
    x->objectName = gensym(objectName);
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
    PyObject *nooutlet = PyDict_GetItemString(PdDict, "py4pdOBJnooutlet");
    int nooutlet_int = PyLong_AsLong(nooutlet);
    x->outPyPointer = PyLong_AsLong(pyOUT);
    x->function_called = 1;
    x->function = pyFunction;
    x->pdPatchFolder = patch_dir;         // set name of the home path
    x->pkgPath = patch_dir;     // set name of the packages path
    x->py_arg_numbers = 0;
    set_py4pd_config(x);  // set the config file  TODO: I want to rethink this)
    py4pd_tempfolder(x);  // find the py4pd temp folder
    findpy4pd_folder(x);  // find the py4pd object folder
    // Parse args for translation between Pd and Python
    PyCodeObject *code = (PyCodeObject*)PyFunction_GetCode(pyFunction);
    x->function_name = gensym(PyUnicode_AsUTF8(code->co_name));
    x->script_name = gensym(PyUnicode_AsUTF8(code->co_filename));
    int parseArgsRight = libraries_parser_args(x, code, argc, argv); // NOTE: added
    if (parseArgsRight == 0) {
        return NULL;
    }
    create_pyObject_inlets(x, argc, argv);
    if (nooutlet_int == 0){
        x->out1 = outlet_new(&x->x_obj, &s_signal);
    }
    // check if numpy array is imported
    int numpyArrayImported = _import_array();
    if (numpyArrayImported == 0) {
        x->numpyImported = 1;
    }
    else{
        x->numpyImported = 0;
        pd_error(x, "[py4pd] Not possible to import numpy array");
        return NULL;
    }
    
    object_count++;
    return (x);
}

// =====================================
void *New_Audio_Object(t_symbol *s, int argc, t_atom *argv) {
    const char *objectName = s->s_name;
    t_py *x = (t_py *)pd_new(pyNewObject_Audio);
    x->visMode  = 0;
    x->pyObject = 1;
    x->audioOutput = 1;
    x->audioInput = 1;
    x->x_canvas = canvas_getcurrent();       // pega o canvas atual
    t_canvas *c = x->x_canvas;               // get the current canvas
    t_symbol *patch_dir = canvas_getdir(c);  // directory of opened patch
    char py4pd_objectName[MAXPDSTRING];
    sprintf(py4pd_objectName, "py4pd_ObjectDict_%s", objectName);
    x->objectName = gensym(objectName);
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
    PyObject *nooutlet = PyDict_GetItemString(PdDict, "py4pdOBJnooutlet");
    int nooutlet_int = PyLong_AsLong(nooutlet);
    x->outPyPointer = PyLong_AsLong(pyOUT);
    x->function_called = 1;
    x->function = pyFunction;
    x->pdPatchFolder = patch_dir;         // set name of the home path
    x->pkgPath = patch_dir;     // set name of the packages path
    x->py_arg_numbers = 0;
    set_py4pd_config(x);  // set the config file  TODO: I want to rethink this)
    py4pd_tempfolder(x);  // find the py4pd temp folder
    findpy4pd_folder(x);  // find the py4pd object folder
    // Parse args for translation between Pd and Python
    PyCodeObject *code = (PyCodeObject*)PyFunction_GetCode(pyFunction);
    x->function_name = gensym(PyUnicode_AsUTF8(code->co_name));
    x->script_name = gensym(PyUnicode_AsUTF8(code->co_filename));
    int parseArgsRight = libraries_parser_args(x, code, argc, argv); // NOTE: added
    if (parseArgsRight == 0) {
        return NULL;
    }
    create_pyObject_inlets(x, argc, argv);
    if (nooutlet_int == 0){
        x->out1 = outlet_new(&x->x_obj, &s_signal);
    }
    // check if numpy array is imported
    int numpyArrayImported = _import_array();
    if (numpyArrayImported == 0) {
        x->numpyImported = 1;
    }
    else{
        x->numpyImported = 0;
        pd_error(x, "[py4pd] Not possible to import numpy array");
        return NULL;
    }
    object_count++;
    return (x);
}


// =====================================
void *pyObjectFree(t_py *x) {
    if (object_count == 0) {
        // Py_Finalize(); // BUG: This not work properly with submodules written in C
        // post("[py4pd] Trying to finalize python");
        object_count = 0;
        #ifdef _WIN64
            char command[1000];
            sprintf(command, "del /q /s %s\\*", x->tempPath->s_name);
            SHELLEXECUTEINFO sei = {0};
            sei.cbSize = sizeof(SHELLEXECUTEINFO);
            sei.fMask = SEE_MASK_NOCLOSEPROCESS;
            sei.lpFile = "cmd.exe";
            sei.lpParameters = command;
            sei.nShow = SW_HIDE;
            ShellExecuteEx(&sei);
            CloseHandle(sei.hProcess);
        #else
            char command[1000];
            sprintf(command, "rm -rf %s", x->tempPath->s_name);
            system(command);
        #endif
    }

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
    PyObject *Function, *showFunction;
    int w = 250, h = 250;
    int objpyout = 0;
    int nooutlet = 0;
    int added2pd_info = 0;
    showFunction = NULL;

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
        /* NOTE: Version 0.8.0
        if (PyDict_Contains(keywords, PyUnicode_FromString("showFunction"))){
            post("[Python]: showFunction");
            showFunction = PyDict_GetItemString(keywords, "showFunction"); // it gets the data type output
        }
        */
    }
    // Add configs to the object
    PyObject *nestedDict = PyDict_New();
    PyDict_SetItemString(nestedDict, "py4pdOBJFunction", Function);
    // PyDict_SetItemString(nestedDict, "py4pdOBJshowFunction", showFunction);
    PyDict_SetItemString(nestedDict, "py4pdOBJwidth", PyLong_FromLong(w));
    PyDict_SetItemString(nestedDict, "py4pdOBJheight", PyLong_FromLong(h));
    PyDict_SetItemString(nestedDict, "py4pdOBJpyout", PyLong_FromLong(objpyout));
    PyDict_SetItemString(nestedDict, "py4pdOBJnooutlet", PyLong_FromLong(nooutlet));
    PyDict_SetItemString(nestedDict, "py4pdOBJname", PyUnicode_FromString(objectName));
    PyObject *objectDict = PyDict_New();
    PyDict_SetItemString(objectDict, objectName, nestedDict);
    PyObject *py4pd_capsule = PyCapsule_New(objectDict, objectName, NULL);
    char py4pd_objectName[MAXPDSTRING];
    sprintf(py4pd_objectName, "py4pd_ObjectDict_%s", objectName);
    PyModule_AddObject(PyImport_ImportModule("__main__"), py4pd_objectName, py4pd_capsule);
    // =====================================
    PyCodeObject* code = (PyCodeObject*)PyFunction_GetCode(Function);
    int py_args = code->co_argcount;

    // NORMAL
    if ((strcmp(objectType, "NORMAL") == 0)){
        pyNewObject = class_new(gensym(objectName), (t_newmethod)New_NORMAL_Object, (t_method)pyObjectFree, sizeof(t_py), CLASS_DEFAULT, A_GIMME, 0);
        class_addmethod(pyNewObject, (t_method)py_Object, gensym("PyObject"), A_POINTER, 0);
        class_addmethod(pyNewObject, (t_method)documentation, gensym("doc"), 0, 0);
        class_addmethod(pyNewObject, (t_method)set_param, gensym("key"), A_GIMME, 0);
        class_addmethod(pyNewObject, (t_method)usepointers, gensym("pointers"), A_FLOAT, 0);
        class_addmethod(pyNewObject, (t_method)reloadObject, gensym("reload"), 0, 0);
        class_addanything(pyNewObject, py_anything);
    }
    // VIS
    else if ((strcmp(objectType, "VIS") == 0)){
        pyNewObject_VIS = class_new(gensym(objectName), (t_newmethod)New_VIS_Object, (t_method)pyObjectFree, sizeof(t_py), CLASS_DEFAULT, A_GIMME, 0);
        class_addanything(pyNewObject_VIS, py_anything);
        class_addmethod(pyNewObject_VIS, (t_method)PY4PD_zoom, gensym("zoom"), A_CANT, 0);
        class_addmethod(pyNewObject_VIS, (t_method)py_Object, gensym("PyObject"), A_POINTER, 0);
        class_addmethod(pyNewObject_VIS, (t_method)documentation, gensym("doc"), 0, 0);
        class_addmethod(pyNewObject_VIS, (t_method)reloadObject, gensym("reload"), 0, 0);
        class_setsavefn(pyNewObject_VIS, &py4pdObjPic_save);
    }
    // AUDIOIN
    else if ((strcmp(objectType, "AUDIOIN") == 0)){
        pyNewObject_AudioIn = class_new(gensym(objectName), (t_newmethod)New_AudioIN_Object, (t_method)pyObjectFree, sizeof(t_py), CLASS_DEFAULT, A_GIMME, 0);
        class_addmethod(pyNewObject_AudioIn, (t_method)documentation, gensym("doc"), 0, 0);
        class_addmethod(pyNewObject_AudioIn, (t_method)reloadObject, gensym("reload"), 0, 0);
        class_addmethod(pyNewObject_AudioIn, (t_method)library_dsp, gensym("dsp"), A_CANT, 0);  // add a method to a class
        CLASS_MAINSIGNALIN(pyNewObject_AudioIn, t_py, py4pdAudio);
    }
    // AUDIOIN
    else if ((strcmp(objectType, "AUDIOOUT") == 0)){
        pyNewObject_AudioOut = class_new(gensym(objectName), (t_newmethod)New_AudioOUT_Object, (t_method)pyObjectFree, sizeof(t_py), CLASS_DEFAULT, A_GIMME, 0);
        class_addmethod(pyNewObject_AudioOut, (t_method)documentation, gensym("doc"), 0, 0);
        class_addmethod(pyNewObject_AudioOut, (t_method)reloadObject, gensym("reload"), 0, 0);
        class_addmethod(pyNewObject_AudioOut, (t_method)library_dsp, gensym("dsp"), A_CANT, 0);  // add a method to a class
        class_addanything(pyNewObject_AudioOut, py_anything);
        // CLASS_MAINSIGNALIN(pyNewObject_AudioOut, t_py, py4pdAudio);
    }
    // AUDIO
    else if (strcmp(objectType, "AUDIO") == 0) {
        pyNewObject_Audio = class_new(gensym(objectName), (t_newmethod)New_Audio_Object, (t_method)pyObjectFree, sizeof(t_py), CLASS_DEFAULT, A_GIMME, 0);
        class_addmethod(pyNewObject_Audio, (t_method)documentation, gensym("doc"), 0, 0);
        class_addmethod(pyNewObject_Audio, (t_method)reloadObject, gensym("reload"), 0, 0);
        class_addmethod(pyNewObject_Audio, (t_method)library_dsp, gensym("dsp"), A_CANT, 0);  // add a method to a class
        CLASS_MAINSIGNALIN(pyNewObject_Audio, t_py, py4pdAudio);
    }
    else{
        // set py error
        PyErr_SetString(PyExc_TypeError, "Object type not supported, check the spelling");
        return NULL;
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
