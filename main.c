#include <Python.h>
#include <m_pd.h>


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
    Py_Initialize();
       
    // get directory of python 3 and print it
    Py_SetPythonHome("C:/Users/Neimog/Git/py-from-C/");
    
}

// ============================================

static void py_run(t_py *x, t_symbol *s, int argc, t_atom *argv){
    // int ;
    PyObject *pName, *pModule, *pDict, *pFunc;
    PyObject *pArgs, *pValue;
    int i;

    if (argc < 3) {
        fprintf(stderr,"Usage: call pythonfile funcname [args]\n");
        return;
    }
    // "C:/Users/Neimog/Git/py-from-C/"
    // Py_SetPythonHome();
    Py_Initialize();
    // define ps = PyUnicode_FromString("C:/Users/Neimog/Git/py-from-C/");
    
    PyObject *pobj = PySys_GetObject(const_cast<char *>("path"));
    PyObject *ps;
    ps = PyUnicode_FromString("C:/Users/Neimog/Git/py-from-C/");


    // ============== DEBUG ============== 
    
    t_symbol *script_file_name = atom_gensym(argv+0);
    post("Script Name = %s.", script_file_name->s_name);

    t_symbol *function_name = atom_gensym(argv+1);
    post("Function Name = %s.", function_name->s_name);

    t_symbol *functions_args = atom_gensym(argv+2);
    post("Function Args = %s.", functions_args->s_name);

    // ============== DEBUG ============== 

    // script_file_name to char
    char *script_file_name_char = script_file_name->s_name;
    
    pName = PyUnicode_DecodeFSDefault(script_file_name_char); // Esse é o nome do script Python
    pModule = PyImport_Import(pName); 
    Py_DECREF(pName);
    post("pModule == %s.", pModule)
    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, function_name->s_name);
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(argc - 2);
            for (i = 0; i < argc - 2; ++i) {
                
                pValue = PyFloat_FromDouble(atom_getfloat(argv+i));
                //
                // print pValue
                post("pValue = %f", atom_getfloat(argv+i));
                
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

void py2pd_free(t_py *x){
    outlet_free(x->out_A);
    pd_free((t_pd *)x);
}

// ====================================================
void py2pd_setup(void){
    py_class =     class_new(gensym("py2pd"), // cria o objeto quando escrevemos py2pd
                        (t_newmethod)py_new, // o methodo de inicializacao | pointer genérico
                        (t_method)py2pd_free, // quando voce deleta o objeto
                        sizeof(t_py), // quanta memoria precisamos para esse objeto
                        CLASS_DEFAULT, // nao há uma GUI especial para esse objeto
                        0); // todos os outros argumentos por exemplo um numero seria A_DEFFLOAT
    
    // class_addmethod(py_class, (t_method)pycontrol_dir, gensym("path"), A_DEFFLOAT, A_DEFSYMBOL, 0);
    class_addmethod(py_class, (t_method)py_run, gensym("run"), A_GIMME, 0); 
    class_addmethod(py_class, (t_method)py_home, gensym("home"), 0);
    }
