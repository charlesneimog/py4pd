#include <Python.h>
#include <m_pd.h>

// DEFINE GLOBAL VARIABLES DEBUG

static t_class *py_class;

// ============================================
typedef struct _py { // It seems that all the objects are some kind of class.
    t_object        x_obj; // convensao no puredata source code
    PyObject        *module; // python object
    PyObject        *function; // function name
    t_symbol        *name; // function name
    t_outlet        *out_A; // outlet 1.
}t_py;

// ====================================

static void py_home(t_py *x) {
    post("Not implemented yet");
    
}

// ====================================

static void py_set_function(t_py *x, t_symbol *s, int argc, t_atom *argv){
    PyObject *pName, *pModule, *pDict, *pFunc;
    PyObject *pArgs, *pValue;

    if (argc < 2) { // check is the number of arguments is correct
        pd_error(x,"py4pd | run method | missing arguments");
        return;
    }
    wchar_t py_name[5];
    wchar_t *py_name_ptr = py_name;
    py_name_ptr = "py4pd";
    Py_SetProgramName(py_name_ptr); 
    Py_Initialize();
    Py_GetPythonHome();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('C:/Users/Neimog/Git/py4pd')"); // TODO: set path using where pd patch is located

    // site modules are in the site-packages folder
    
    PyObject* sys = PyImport_ImportModule( "sys" );
    PyObject* sys_path = PyObject_GetAttrString( sys, "path" );
    PyObject* folder_path = PyUnicode_FromString("C:/Users/Neimog/Documents/OM#/temp-files/OM-py-env/Lib/site-packages/");
    PyList_Append( sys_path, folder_path);
    
    // ============================================================
    
    t_symbol *script_file_name = atom_gensym(argv+0);
    t_symbol *function_name = atom_gensym(argv+1);
        
    pName = PyUnicode_DecodeFSDefault(script_file_name->s_name); // Esse é o nome do script Python
    pModule = PyImport_Import(pName);
    pFunc = PyObject_GetAttrString(pModule, function_name->s_name); // Esse é o nome da função Python 
    Py_DECREF(pName);
    if (pFunc && PyCallable_Check(pFunc)){ // Verifica se a função existe
        // pFunc equal x_function
        x->function = pFunc;
        x->module = pModule;
        post("py4pd | function '%s' loaded!", function_name->s_name);
        return;
    } else {
        // post PyErr_Print() in pd
        pd_error(x, "py4pd | function %s not loaded!", function_name->s_name);
        post("\n===== PYTHON ERROR =====");
        // show python error in pd using post
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        post("%s", PyUnicode_AsUTF8(pstr));
        post("========================\n");
        return;
    }
}

// ============================================

static void py_run_without_quit_py(t_py *x, t_symbol *s, int argc, t_atom *argv){
    PyObject *pName, *pFunc; // pDict, *pModule,
    PyObject *pArgs, *pValue;
    pFunc = x->function;
    // pModule = x->module;
    pArgs = PyTuple_New(argc);
    int i;
    for (i = 0; i < argc; ++i) {
        if (argv[i].a_type == A_FLOAT) {
            pValue = PyFloat_FromDouble(argv[i+0].a_w.w_float);
            // post("py4pd | run method | float: %f", argv[i+0].a_w.w_float);
        } else if (argv[i].a_type == A_SYMBOL) {
            pValue = PyUnicode_DecodeFSDefault(argv[i].a_w.w_symbol->s_name);
            // post("py4pd | run method | symbol: %s", argv[i].a_w.w_symbol->s_name);
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
    
    // Py_DECREF(pArgs);
    
    if (pValue != NULL) {
        Py_DECREF(pValue);
        outlet_float(x->out_A, PyLong_AsLong(pValue));
    }
    else {
        PyErr_Print();
        pd_error(x,"Call failed\n");
        return;
    }

}


// ============================================
// ============================================
// ============================================
// ============================================
// ============================================
// ============================================
// ============================================
// ============================================
// ============================================
// ============================================

static void py_run(t_py *x, t_symbol *s, int argc, t_atom *argv){

    PyObject *pName, *pModule, *pDict, *pFunc;
    PyObject *pArgs, *pValue;

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
    PyRun_SimpleString("import sys");
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
// ============================================
// ============================================

void *py_new(void){
    t_py *x = (t_py *)pd_new(py_class); // pointer para a classe

    // set name 
    char *name = "py4pd";
    x->name = gensym(name);
    x->out_A = outlet_new(&x->x_obj, &s_anything); // cria um outlet
    return(x);
}

// ============================================
// ============================================
// ============================================

void py4pd_free(t_py *x){
    PyObject  *pModule, *pFunc; // pDict, *pName,
    pFunc = x->function;
    pModule = x->module;
    Py_DECREF(pFunc);
    Py_DECREF(pModule);
    Py_FinalizeEx();
    // outlet_free(x->out_A);
    // pd_free((t_pd *)x);
}

// ====================================================
void py4pd_setup(void){
    py_class =     class_new(gensym("py4pd"), // cria o objeto quando escrevemos py4pd
                        (t_newmethod)py_new, // o methodo de inicializacao | pointer genérico
                        (t_method)py4pd_free, // quando voce deleta o objeto
                        sizeof(t_py), // quanta memoria precisamos para esse objeto
                        CLASS_DEFAULT, // nao há uma GUI especial para esse objeto
                        0); // todos os outros argumentos por exemplo um numero seria A_DEFFLOAT
    
    // class_addmethod(py_class, (t_method)pycontrol_dir, gensym("path"), A_DEFFLOAT, A_DEFSYMBOL, 0);
    class_addmethod(py_class, (t_method)py_run, gensym("run"), A_GIMME, 0); 
    class_addmethod(py_class, (t_method)py_home, gensym("home"), 0);
    class_addmethod(py_class, (t_method)py_set_function, gensym("set"), A_GIMME, 0);
    class_addmethod(py_class, (t_method)py_run_without_quit_py, gensym("args"), A_GIMME, 0);
    // class_addmethod(py_class, (t_method)py_import, gensym("import"), 0, 0);
    }