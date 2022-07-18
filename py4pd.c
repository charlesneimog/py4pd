// =================================
// Exact copy of https://github.com/pure-data/pure-data/src/x_gui.c 
#include <m_pd.h>
#include <g_canvas.h>
#include <stdio.h>
#include <string.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef _MSC_VER
#define snprintf _snprintf  /* for pdcontrol object */
#endif
// end of the copy of x_gui.c

#include <Python.h>

// =================================
// ============ Pd Object code  ====
// =================================

static t_class *py_class;

// =====================================
typedef struct _py { // It seems that all the objects are some kind of class.
    t_object        x_obj; // convensao no puredata source code
    t_canvas        *x_canvas; // pointer to the canvas
    PyObject        *module; // python object
    PyObject        *function; // function name
    t_float         *set_was_called; // flag to check if the set function was called
    t_symbol        *packages_path; // packages path 
    t_symbol        *home_path; // home path this always is the path folder (?)
    t_symbol        *function_name; // function name
    t_outlet        *out_A; // outlet 1.
}t_py;

// ============================================
// ============== METHODS =====================
// ============================================

static void home(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc < 1) {
        post("The home path is: %s", x->home_path->s_name);
        return; // is this necessary?
    }
    // TODO: make it define others paths 
    // TODO: make it work with path with spaces
}

// ============================================
// ============================================
// ============================================

static void packages(t_py *x, t_symbol *s, int argc, t_atom *argv) {

    if (argc < 1) {
        post("The packages path is: %s", x->packages_path->s_name);
        return; // is this necessary?
    }
    else {
        if (argc < 2 && argc > 0){
            x->packages_path = atom_getsymbol(argv);
            post("The packages path is now: %s", x->packages_path->s_name);
        }   
        else
            pd_error(x, "It seems that your package folder has |spaces|. It can not have |spaces|!");
            post("I intend to implement this feature in the future!");
            return;
    }
}

// ====================================
// ====================================
// ====================================

static void set_function(t_py *x, t_symbol *s, int argc, t_atom *argv){
    t_symbol *script_file_name = atom_gensym(argv+0);
    t_symbol *function_name = atom_gensym(argv+1);
    
    if (x->function_name != NULL){
        int function_is_equal = strcmp(function_name->s_name, x->function_name->s_name);
        if (function_is_equal == 0){    // If the user wants to call the same function again! This is not necessary at first glance. 
            pd_error(x, "WARNING :: The function was already called!");
            pd_error(x, "WARNING :: Calling the function again! This make it slower!");
            post("");
            return;
        }
        else{ // If the function is different, then we need to delete the old function and create a new one.
            Py_XDECREF(x->function);
            Py_XDECREF(x->module);
            x->function = NULL;
            x->module = NULL;
            x->function_name = NULL;
        }      
    }

    // =====================

    if (argc < 2) { // check is the number of arguments is correct | set "function_script" "function_name"
        pd_error(x,"py4pd :: The set message needs two arguments! The 'Script name' and the 'function name'!");
        return;
    }
    
    PyObject *pName, *pModule, *pDict, *pFunc,  *py_func_obj=NULL;
    PyObject *pArgs, *pValue;
    wchar_t py_name[5];
    wchar_t *py_name_ptr = py_name;
    py_name_ptr = "py4pd";
    Py_SetProgramName(py_name_ptr); // set program name
    Py_Initialize();
    Py_GetPythonHome();

    // =====================
    char *home_path_str = x->home_path->s_name;
    char *sys_path_str = malloc(strlen(home_path_str) + strlen("sys.path.append('") + strlen("')") + 1);
    sprintf(sys_path_str, "sys.path.append('%s')", home_path_str);
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(sys_path_str);
    free(sys_path_str); // free the memory allocated for the string, check if this is necessary
    
    // =====================

    char *site_path_str = x->packages_path->s_name;

    PyObject* sys = PyImport_ImportModule("sys");
    PyObject* sys_path = PyObject_GetAttrString(sys, "path");
    // TODO: make possible to set own site-packages path
    PyObject* folder_path = PyUnicode_FromString("C:/Users/Neimog/Documents/OM#/temp-files/OM-py-env/Lib/site-packages"); // 
    // TODO: make possible to set own site-packages path
    PyList_Append(sys_path, folder_path);
    Py_DECREF(folder_path);
    Py_DECREF(sys_path);
    
    // ============================================================
    
    pName = PyUnicode_DecodeFSDefault(script_file_name->s_name); // Name of script file
    pModule = PyImport_Import(pName);
    pFunc = PyObject_GetAttrString(pModule, function_name->s_name); // Name of the Function name inside the script file
    Py_DECREF(pName);
    if (pFunc && PyCallable_Check(pFunc)){ // Check if the function exists and is callable
        // pFunc equal x_function
        x->function = pFunc;
        x->module = pModule;
        post("py4pd | function '%s' loaded!", function_name->s_name);
        x->function_name = function_name;
        x->set_was_called = 1;
        return;
    } else {
        // post PyErr_Print() in pd
        pd_error(x, "py4pd | function %s not loaded!", function_name->s_name);
        x->set_was_called = 0; // set the flag to 0 because it crash Pd if user try to use args method
        x->function_name = NULL;
        post("");
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "Call failed:\n %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        return;
    }
}

// ============================================
// ============================================
// ============================================

static void run_without_quit_py(t_py *x, t_symbol *s, int argc, t_atom *argv){
    if (x->set_was_called == 0) {
        pd_error(x, "You need to send a message ||| set {script_file_name} {function_name} ||| first!");
        // TODO: after use run method, create another message.
        return;
    }
    
    PyObject *pName, *pFunc; // pDict, *pModule,
    PyObject *pArgs, *pValue;
    pFunc = x->function;
    pArgs = PyTuple_New(argc);
    int i;
    for (i = 0; i < argc; ++i) {
        if (argv[i].a_type == A_FLOAT) {
            pValue = PyFloat_FromDouble(argv[i+0].a_w.w_float); // convert to python float
        } else if (argv[i].a_type == A_SYMBOL) {
            pValue = PyUnicode_DecodeFSDefault(argv[i].a_w.w_symbol->s_name); // convert to python string
        } else {
            pValue = Py_None;
            Py_INCREF(Py_None);
        }
                
        if (!pValue) {
            pd_error(x, "Cannot convert argument\n");
            return;
        }
        
        PyTuple_SetItem(pArgs, i, pValue);
    }

    pValue = PyObject_CallObject(pFunc, pArgs);
    if (pValue != NULL) {
        outlet_float(x->out_A, PyLong_AsLong(pValue)); // TODO: make it iterate with more types
        Py_DECREF(pValue);
    }
    else {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        return;
    }
}

// ============================================
// ============================================
// ============================================

static void run(t_py *x, t_symbol *s, int argc, t_atom *argv){

    PyObject *pName, *pModule, *pDict, *pFunc;
    PyObject *pArgs, *pValue;
    
    if (x->function_name != NULL){
        Py_XDECREF(x->function); // free the memory allocated for the function
        Py_XDECREF(x->module); // free the memory allocated for the module
        x->function = NULL; // clear the function pointer
        x->module = NULL; // clear the module pointer
        x->function_name = NULL; // clear the function name pointer
        x->set_was_called = 0; // set the flag to 0 because it crash Pd if user try to use args method without set first
    }
    
    if (argc < 3) {
        pd_error(x,"py4pd | run method | missing arguments");
        return;
    }

    wchar_t py_name[5];
    wchar_t *py_name_ptr = py_name;
    py_name_ptr = "py4pd";
    Py_SetProgramName(py_name_ptr);  
    Py_Initialize();
    Py_GetPythonHome();
    PyRun_SimpleString("import sys"); // TODO: make this like the set method
    PyRun_SimpleString("sys.path.append('C:/Users/Neimog/Git/py4pd')"); // set path using where pd patch is located

    t_symbol *script_file_name = atom_gensym(argv+0);
    t_symbol *function_name = atom_gensym(argv+1);
        
    pName = PyUnicode_DecodeFSDefault(script_file_name->s_name); // Esse é o nome do script Python
    pModule = PyImport_Import(pName); 
    Py_DECREF(pName);

    int i;
    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, function_name->s_name);
        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(argc - 2);
            for (i = 0; i < argc - 2; ++i) {
                if (argv[i+2].a_type == A_FLOAT) {
                    pValue = PyFloat_FromDouble(argv[i+2].a_w.w_float);
                } else if (argv[i+2].a_type == A_SYMBOL) {
                    pValue = PyUnicode_DecodeFSDefault(argv[i+2].a_w.w_symbol->s_name);
                } else {
                    pValue = Py_None;
                    Py_INCREF(Py_None);
                }
                      
                if (!pValue) {
                    Py_DECREF(pArgs);
                    Py_DECREF(pModule);
                    pd_error(x, "Cannot convert argument\n");
                    return;
                }
                PyTuple_SetItem(pArgs, i, pValue);
            }

            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                Py_DECREF(pValue);
                outlet_float(x->out_A, PyLong_AsLong(pValue));
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                pd_error(x,"Call failed\n");
                return;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            pd_error(x, "Cannot find function \"%s\"\n", function_name->s_name);
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        pd_error(x, "Failed to load \"%s\"\n", script_file_name->s_name);
        
        return ;
    }
    if (Py_FinalizeEx() < 0) {
        return;
    }
    return;
}

// ============================================
// =========== CREATION OF OBJECT =============
// ============================================

void *py_new(void){
    // credits
    post("");
    post("");
    post("");
    post("py4pd by Charles K. Neimog");
    post("version 0.0         ");
    post("Based on Python 3.10.0  ");
    post("");
    post("It uses code from: Alexandre Porres, Thomas Grill and Miller Puckette and others");
    post("");
    post("");

    // pd things
    t_py *x = (t_py *)pd_new(py_class); // pointer para a classe
    x->x_canvas = canvas_getcurrent(); // pega o canvas atual
    x->out_A = outlet_new(&x->x_obj, &s_anything); // cria um outlet
    // ========

    // py things
    t_canvas *c = x->x_canvas; 
    x->home_path = canvas_getdir(c);     // set name 
    x->packages_path = canvas_getdir(c); // set name
    return(x);
}

// ============================================
// =========== REMOVE OBJECT ==================
// ============================================

void py4pd_free(t_py *x){
    PyObject  *pModule, *pFunc; // pDict, *pName,
    pFunc = x->function;
    pModule = x->module;
    if (pModule != NULL) {
        Py_DECREF(pModule);
    }
    if (pFunc != NULL) {
        Py_DECREF(pFunc);
    }
    if (Py_FinalizeEx() < 0) {
        return;
    }
}

// ====================================================
void py4pd_setup(void){
    py_class =     class_new(gensym("py4pd"), // cria o objeto quando escrevemos py4pd
                        (t_newmethod)py_new, // o methodo de inicializacao | pointer genérico
                        (t_method)py4pd_free, // quando voce deleta o objeto
                        sizeof(t_py), // quanta memoria precisamos para esse objeto
                        CLASS_DEFAULT, // nao há uma GUI especial para esse objeto
                        0); // todos os outros argumentos por exemplo um numero seria A_DEFFLOAT
    
    class_addmethod(py_class, (t_method)run, gensym("run"), A_GIMME, 0); 
    class_addmethod(py_class, (t_method)home, gensym("home"), A_GIMME, 0);
    class_addmethod(py_class, (t_method)set_function, gensym("set"), A_GIMME, 0);
    class_addmethod(py_class, (t_method)run_without_quit_py, gensym("args"), A_GIMME, 0); // TODO: better name for this method
    class_addmethod(py_class, (t_method)packages, gensym("packages"), A_GIMME, 0);
    }