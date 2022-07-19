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
        x->set_was_called = 2; // set the flag to 0 because it crash Pd if user try to use args method without set first
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