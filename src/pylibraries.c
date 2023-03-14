#include "pylibraries.h"
#include "m_pd.h"
#include "py4pd.h"
#include "py4pd_utils.h"

static t_class *pyNewObject;


// =====================================
void py_anything(t_py *x, t_symbol *s, int ac, t_atom *av){
    // check if t_symbol have string "addInlet"
    const char *symbolName = s->s_name;
    if (strstr(symbolName, "addInlet") != NULL) {
        // get the number of inlet
        post("save inlet");
    }
    else{
        post("run function");
    }




    return;
}

// =====================================
void *py_newObject(t_symbol *s, int argc, t_atom *argv) {
    const char *objectName = s->s_name;
    t_py *x = (t_py *)pd_new(pyNewObject);


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

    // create inlet
    for (i = 0; i < x->py_arg_numbers - 1; i++){
        char methodForInlet[MAXPDSTRING];
        sprintf(methodForInlet, "addInlet %d", i + 1);
        inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_list, gensym(methodForInlet));
    }

    x->out_A = outlet_new(&x->x_obj, 0);

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

    // add home_path
    pyNewObject = class_new(gensym(objectName), (t_newmethod)py_newObject, 0, sizeof(t_py), CLASS_DEFAULT, A_GIMME, 0);
    class_addanything(pyNewObject, py_anything);
    
    return PyLong_FromLong(0);
}
