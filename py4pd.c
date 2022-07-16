#include <Python.h>
#include <m_pd.h>

// DEFINE GLOBAL VARIABLES DEBUG

static t_class *py_class;

// ============================================
typedef struct _py { // It seems that all the objects are some kind of class.
    t_object        x_obj; // convensao no puredata source
    t_symbol        *x_pyvariables; // array variables???
    t_symbol        *x_pyenvironment; // function name
    t_symbol        *x_pyscript; // symbol of the function
    t_float         *debug; // float value
    t_canvas        *x_canvas; // canvas
    t_outlet        *out_A; // outlet 1.
}t_py;

// ====================================

static void py_home(t_py *x) {
    post("Not implemented yet");
    
}

// ============================================

static void py_run(t_py *x, t_symbol *s, int argc, t_atom *argv){
    // int ;
    PyObject *pName, *pModule, *pDict, *pFunc;
    PyObject *pArgs, *pValue;
    // int i;

    if (argc < 3) {
        fprintf(stderr,"Usage: call pythonfile funcname [args]\n");
        return;
    }
    
    Py_SetProgramName("py4pd"); 
    Py_Initialize();

    Py_GetPythonHome();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('C:/Users/Neimog/Git/py4pd')");
    
    // PyRun_SimpleString("sys.path.append(os.getcwd())");

    t_symbol *script_file_name = atom_gensym(argv+0);
    t_symbol *function_name = atom_gensym(argv+1);
    t_symbol *functions_args = atom_gensym(argv+2);
    
    // ============== DEBUG ============== 
    if (0 == 1) {
        post("Script Name = %s.", script_file_name->s_name);
        post("Function Name = %s.", function_name->s_name);
        post("Function Args = %s.", functions_args->s_name);
    }
    
    // ============== DEBUG ============== 

    char *script_file_name_char = script_file_name->s_name;
    
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
                    // post("Argument %d is Float = %f.", i, argv[i+2].a_w.w_float);
                } else if (argv[i+2].a_type == A_SYMBOL) {
                    pValue = PyUnicode_DecodeFSDefault(argv[i+2].a_w.w_symbol->s_name);
                    // post("Argument %d is Symbol = %s.", i, argv[i+2].a_w.w_symbol->s_name);
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
                /* pValue reference stolen here: */
                PyTuple_SetItem(pArgs, i, pValue);
            }
            
            
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            // post("Return value: %s", pValue);
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
void *py_new(void){
    t_py *x = (t_py *)pd_new(py_class); // pointer para a classe
    x->out_A = outlet_new(&x->x_obj, &s_anything); // cria um outlet
    return(x);
}

// ============================================

void py4pd_free(t_py *x){
    outlet_free(x->out_A);
    pd_free((t_pd *)x);
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
    }