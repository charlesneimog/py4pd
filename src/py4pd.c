#include "py4pd.h"
#include "m_pd.h"
#include "module.h"

// ===================================================================
// ========================= Utilities ===============================
// ===================================================================


/* \brief Convert the pd object to python object
 * \param pd_value Pointer to the value from pd
 * \return Pointer to the python object
 */

static PyObject *py4pd_convert_to_python(t_atom *pd_value) {
    PyObject *pValue;
    if (pd_value->a_type == A_FLOAT){ 
            float arg_float = atom_getfloat(pd_value);
            if (arg_float == (int)arg_float){ // DOC: If the float is an integer, then convert to int
                int arg_int = (int)arg_float;
                pValue = PyLong_FromLong(arg_int);
            }
            else{ // If the int is an integer, then convert to int
                pValue = PyFloat_FromDouble(arg_float);
            }
    } else if (pd_value->a_type == A_SYMBOL) {
        pValue = PyUnicode_DecodeFSDefault(pd_value->a_w.w_symbol->s_name);
    } else {
        pValue = Py_None;
        Py_INCREF(Py_None);
    }
    return pValue;
}

// =====================================================================
/* \brief Convert the python object to pd object
 * \param x Pointer to the py4pd object
 * \param pValue Pointer to the python object
 * \return void (output the result to the pd object)
 */

static void *py4pd_convert_to_pd(t_py *x, PyObject *pValue) {
    
    if (PyList_Check(pValue)){                       // DOC: If the function return a list list
        int list_size = PyList_Size(pValue);
        t_atom *list_array = (t_atom *) malloc(list_size * sizeof(t_atom));     
        int i;       
        for (i = 0; i < list_size; ++i) {
            PyObject *pValue_i = PyList_GetItem(pValue, i);
            if (PyLong_Check(pValue_i)) {            // DOC: If the function return a list of integers
                long result = PyLong_AsLong(pValue_i);
                float result_float = (float)result;
                list_array[i].a_type = A_FLOAT;
                list_array[i].a_w.w_float = result_float;

            } else if (PyFloat_Check(pValue_i)) {    // DOC: If the function return a list of floats
                double result = PyFloat_AsDouble(pValue_i);
                float result_float = (float)result;
                list_array[i].a_type = A_FLOAT;
                list_array[i].a_w.w_float = result_float;
            } else if (PyUnicode_Check(pValue_i)) {  // DOC: If the function return a list of strings
                const char *result = PyUnicode_AsUTF8(pValue_i); 
                list_array[i].a_type = A_SYMBOL;
                list_array[i].a_w.w_symbol = gensym(result);
            } else if (Py_IsNone(pValue_i)) {        // DOC: If the function return a list of None
            } else {
                pd_error(x, "[py4pd] py4pd just convert int, float and string! Received: %s", Py_TYPE(pValue_i)->tp_name);
                Py_DECREF(pValue_i);
                return 0;
            }
        }
        outlet_list(x->out_A, 0, list_size, list_array); // TODO: possible do in other way? Seems slow!
        return 0;
    } else {
        if (PyLong_Check(pValue)) {
            long result = PyLong_AsLong(pValue); // DOC: If the function return a integer
            outlet_float(x->out_A, result);
            //PyGILState_Release(gstate);
            return 0;
        } else if (PyFloat_Check(pValue)) {
            double result = PyFloat_AsDouble(pValue); // DOC: If the function return a float
            float result_float = (float)result;
            outlet_float(x->out_A, result_float);
            //PyGILState_Release(gstate);
            return 0;
            // outlet_float(x->out_A, result);
        } else if (PyUnicode_Check(pValue)) {
            const char *result = PyUnicode_AsUTF8(pValue); // DOC: If the function return a string
            outlet_symbol(x->out_A, gensym(result)); 
            return 0;
            
        } else if (Py_IsNone(pValue)) {
        } else {
            pd_error(x, "[py4pd] py4pd just convert int, float and string or list of this atoms! Received: %s", Py_TYPE(pValue)->tp_name);
            return 0;
        }
    }
    return 0;

}

// =====================================================================
// ========================= py4pd object =============================

static int *set_py4pd_config(t_py *x) {
    
    // check if in x->home_path there is a file py4pd.config
    char *config_path = (char *)malloc(sizeof(char) * (strlen(x->home_path->s_name) + strlen("/py4pd.cfg") + 1)); // 
    strcpy(config_path, x->home_path->s_name); // copy string one into the result.
    strcat(config_path, "/py4pd.cfg"); // append string two to the result.
    if (access(config_path, F_OK) != -1) { // check if file exists
        post("[py4pd] py4pd.cfg file found");
        FILE* file = fopen(config_path, "r"); /* should check the result */
        char line[256]; // line buffer
        while (fgets(line, sizeof(line), file)) { // read a line
            if (strstr(line, "packages =") != NULL) { // check if line contains "packages ="
                char *packages_path = (char *)malloc(sizeof(char) * (strlen(line) - strlen("packages = ") + 1)); // 
                strcpy(packages_path, line + strlen("packages = ")); // copy string one into the result.
                
                if (strlen(packages_path) > 0) { // check if path is not empty
                    // from packages_path remove the two last character
                    packages_path[strlen(packages_path) - 1] = '\0'; // remove the last character
                    packages_path[strlen(packages_path) - 1] = '\0'; // remove the last character

                    // remove all spaces from packages_path
                    char *i = packages_path;
                    char *j = packages_path;
                    while(*j != 0) {
                        *i = *j++;
                        if(*i != ' ')
                            i++;
                    }
                    *i = 0;
                    // if packages_path start with . add the home_path
                    if (packages_path[0] == '.') {
                        char *new_packages_path = (char *)malloc(sizeof(char) * (strlen(x->home_path->s_name) + strlen(packages_path) + 1)); // 
                        strcpy(new_packages_path, x->home_path->s_name); // copy string one into the result.
                        strcat(new_packages_path, packages_path + 1); // append string two to the result.
                        x->packages_path = gensym(new_packages_path);
                        free(new_packages_path);
                    } else {
                        x->packages_path = gensym(packages_path);
                    }

                }
                free(packages_path); // free memory
            }
            else if (strstr(line, "thread =") != NULL){
                post("thread");
                char *thread = (char *)malloc(sizeof(char) * (strlen(line) - strlen("thread = ") + 1)); //
                strcpy(thread, line + strlen("thread = ")); // copy string one into the result. TODO: implement thread
                // print thread value
                if (strlen(thread) > 0) { // check if path is not empty
                    // from thread remove the two last character
                    thread[strlen(thread) - 1] = '\0'; // remove the last character
                    thread[strlen(thread) - 1] = '\0'; // remove the last character

                    // remove all spaces from thread
                    char *i = thread;
                    char *j = thread;
                    while(*j != 0) {
                        *i = *j++;
                        if(*i != ' ')
                            i++;
                    }
                    *i = 0;
                    // if thread start with . add the home_path
                    if (thread[0] == '1') {
                        post("value is 1");
                    } else {
                        post("value not 1");
                    }

                }
            }
            else if (strstr(line, "editor =") != NULL){
                pd_error(x, "[py4pd] editor not implemented yet"); // TODO: implement choice for editor (nvim, emacs, code, etc.)
            }


        }
        fclose(file); // close file
    } else {
        x->packages_path = gensym("./py-modules");
    }
    free(config_path); // free memory
    return 0;

}

// ===================================================================
// ========================= Pd Object ===============================
// ===================================================================

static void home(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s; // unused but required by pd
    if (argc < 1) {
        post("[py4pd] The home path is: %s", x->home_path->s_name);
    } else {
        x->home_path = atom_getsymbol(argv);
        post("[py4pd] The home path set to: %s", x->home_path->s_name);
    }
    return;
}

// // ============================================

static void packages(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s; 
    if (argc < 1) {
        post("[py4pd] The packages path is: %s", x->packages_path->s_name);
        return; // is this necessary?
    }
    else {
        if (argc < 2 && argc > 0){
            if (argv[0].a_type == A_SYMBOL) {
                t_symbol *path = atom_getsymbol(argv);

                // It checks relative path

                if (path->s_name[0] == '.' && path->s_name[1] == '/') {
                    char *new_path = malloc(strlen(x->home_path->s_name) + strlen(path->s_name) + 1);
                    strcpy(new_path, x->home_path->s_name);
                    strcat(new_path, path->s_name + 1);
                    post("[py4pd] The packages path set to: %s", new_path);
                    x->packages_path = gensym(new_path);
                    free(new_path);
                } 
                else {
                    x->packages_path = atom_getsymbol(argv);
                    post("[py4pd] The packages path set to: %s", x->packages_path->s_name);
                }
            
            } else{
                pd_error(x, "[py4pd] The packages path must be a string");
                return;
            }
            // check if path exists and is valid
            if (access(x->packages_path->s_name, F_OK) == -1) {
                pd_error(x, "[py4pd] The packages path is not valid");
                return;
            }
        } else{
            pd_error(x, "It seems that your package folder has |spaces|.");
            return;
        }
        return;   
    }
}

// ====================================
// ====================================
// ====================================

static void documentation(t_py *x){
    PyObject *pFunc;
    if (x->function_called == 0) { // if the set method was not called, then we can not run the function :)
        pd_error(x, "[py4pd] To see the documentaion you need to set the function first!");
        return;
    }
    pFunc = x->function;
    if (pFunc && PyCallable_Check(pFunc)){ // Check if the function exists and is callable
        PyObject *pDoc = PyObject_GetAttrString(pFunc, "__doc__"); // Get the documentation of the function
        if (pDoc != NULL){
            const char *Doc = PyUnicode_AsUTF8(pDoc); 
            if (Doc != NULL){
                post("-------- %s --------", x->function_name->s_name);
                post("");
                post("%s", Doc);
                post("");
                return;
            }
            else{
                pd_error(x, "[py4pd] No documentation found!");
                return;
            }
        }
        else{
            pd_error(x, "[py4pd] No documentation found!");
            return;
        }
    }
}

// ====================================
// ====================================
// ====================================

void pd4py_system_func (const char *command){
    int result = system(command);
    if (result == -1){
        post("[py4pd] %s", command);
        return;
    }
}

// ============================================
// ============================================

static void create(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    (void)argc;
    const char *script_name = argv[0].a_w.w_symbol->s_name;
    post("[py4pd] Opening vscode...");

    // DOC: Open VsCode in Windows 
    #ifdef _WIN64 
    char *command = malloc(strlen(x->home_path->s_name) + strlen(script_name) + 20);
    sprintf(command, "/c code %s/%s.py", x->home_path->s_name, script_name);
    SHELLEXECUTEINFO sei = {0};
    sei.cbSize = sizeof(sei);
    sei.fMask = SEE_MASK_NOCLOSEPROCESS;
    sei.lpFile = "cmd.exe ";
    sei.lpParameters = command;
    sei.nShow = SW_HIDE;
    ShellExecuteEx(&sei);
    CloseHandle(sei.hProcess);
    free(command);
    return;

    // DOC: Open VsCode in Linux and Mac

    #else // if not windows 64bits
    char *command = malloc(strlen(x->home_path->s_name) + strlen(script_name) + 20);
    sprintf(command, "code %s/%s.py", x->home_path->s_name, script_name);
    pd4py_system_func(command);
    return;
    #endif
}

// ====================================
// ====================================
// ====================================

static void vscode(t_py *x){
    // If Windows OS run, if not then warn the user
       
    if (x->function_called == 0) { // if the set method was not called, then we can not run the function :)
        pd_error(x, "[py4pd] To open vscode you need to set the function first!");
        return;
    }
    post("[py4pd] Opening vscode...");

    // DOC: Open VsCode in Windows
    #ifdef _WIN64 // ERROR: the endif is missing directive _WIN64
    char *command = malloc(strlen(x->home_path->s_name) + strlen(x->script_name->s_name) + 20);
    sprintf(command, "/c code %s/%s.py", x->home_path->s_name, x->script_name->s_name);
    SHELLEXECUTEINFO sei = {0};
    sei.cbSize = sizeof(sei);
    sei.fMask = SEE_MASK_NOCLOSEPROCESS;
    // sei.lpVerb = "open";
    sei.lpFile = "cmd.exe ";
    sei.lpParameters = command;
    sei.nShow = SW_HIDE;
    ShellExecuteEx(&sei);
    CloseHandle(sei.hProcess);
    return;

    // Not Windows OS
    #else // if not windows 64bits
    char *command = malloc(strlen(x->home_path->s_name) + strlen(x->script_name->s_name) + 20);
    sprintf(command, "code %s/%s.py", x->home_path->s_name, x->script_name->s_name);
    pd4py_system_func(command);
    #endif

    // If macOS
    #ifdef __APPLE__
    pd_error(x, "Not tested in your Platform, please send me a report!");
    #endif
    return ;

}

// ====================================
// ====================================
// ====================================

static void reload(t_py *x){
    PyObject *pName, *pFunc, *pModule, *pReload;
    if (x->function_called == 0) { // if the set method was not called, then we can not run the function :)
        pd_error(x, "To reload the script you need to set the function first!");
        return;
    }
    pFunc = x->function;
    pModule = x->module;

    // reload the module
    pName = PyUnicode_DecodeFSDefault(x->script_name->s_name); // Name of script file
    pModule = PyImport_Import(pName);
    pReload = PyImport_ReloadModule(pModule);
    if (pReload == NULL) {
        pd_error(x, "Error reloading the module!");
        x->function_called = 0;
        Py_DECREF(x->function);
        Py_DECREF(x->module);
        return;
    } else{
        Py_XDECREF(x->module);
        pFunc = PyObject_GetAttrString(pModule, x->function_name->s_name); // Function name inside the script file
        Py_DECREF(pName); // 
        if (pFunc && PyCallable_Check(pFunc)){ // Check if the function exists and is callable 
            x->function = pFunc;
            x->module = pModule;
            x->script_name = x->script_name;
            x->function_name = x->function_name; // why 
            x->function_called = 1; 
            post("The module was reloaded!");
            return; 
        }
        else{
            pd_error(x, "Error reloading the module!");
            x->function_called = 0;
            Py_DECREF(x->function);
            Py_DECREF(x->module);
            return;
        }
    } 
}

// ====================================
// ====================================
// ====================================

static void set_function(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    t_symbol *script_file_name = atom_gensym(argv+0);
    t_symbol *function_name = atom_gensym(argv+1);

    if (x->function_called == 1){
        int function_is_equal = strcmp(function_name->s_name, x->function_name->s_name); // if string is equal strcmp returns 0
        if (function_is_equal == 0){
            pd_error(x, "[py4pd] The function was already set!");
            return;
        }
        else{
            Py_XDECREF(x->function);
            Py_XDECREF(x->module);
            x->function_called = 0;
        }
    }
    
    // DOC: Check if there is extension (not to use it)
    char *extension = strrchr(script_file_name->s_name, '.');
    if (extension != NULL) {
        pd_error(x, "[py4pd] Don't use extensions in the script file name!");
        Py_XDECREF(x->function);
        Py_XDECREF(x->module);
        return;
    }

    // check if script file exists
    char script_file_path[MAXPDSTRING];
    snprintf(script_file_path, MAXPDSTRING, "%s/%s.py", x->home_path->s_name, script_file_name->s_name);
    if (access(script_file_path, F_OK) == -1) {
        pd_error(x, "[py4pd] The script file %s does not exist!", script_file_path);
        Py_XDECREF(x->function);
        Py_XDECREF(x->module);
        return;
    }
    
    // =====================
    // DOC: check number of arguments
    if (argc < 2) { // check is the number of arguments is correct | set "function_script" "function_name"
        pd_error(x,"[py4pd] 'set' message needs two arguments! The 'Script Name' and the 'Function Name'!");
        return;
    }
    // =====================
    PyObject *pName, *pModule, *pFunc; // Create the variables of the python objects
    
    // =====================
    // Add aditional path to python to work with Pure Data
    PyObject *home_path = PyUnicode_FromString(x->home_path->s_name); // DOC: Place where script file will probably be
    PyObject *site_package = PyUnicode_FromString(x->packages_path->s_name); // DOC: Place where the packages will be
    PyObject *sys_path = PySys_GetObject("path");
    PyList_Insert(sys_path, 0, home_path);
    PyList_Insert(sys_path, 0, site_package);
    Py_DECREF(home_path);
    Py_DECREF(site_package);

    // =====================
    pName = PyUnicode_DecodeFSDefault(script_file_name->s_name); // Name of script file
    pModule = PyImport_Import(pName);
    // =====================
    // check if the module was loaded
    if (pModule == NULL) {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_XDECREF(pstr);
        Py_XDECREF(pModule);
        Py_XDECREF(pName);
        return;
    }
 
    pFunc = PyObject_GetAttrString(pModule, function_name->s_name); // Function name inside the script file
    Py_DECREF(pName); // DOC: Delete the name of the script file
    if (pFunc && PyCallable_Check(pFunc)){ // Check if the function exists and is callable   
        PyObject *inspect=NULL, *getfullargspec=NULL, *argspec=NULL, *args=NULL;
        inspect = PyImport_ImportModule("inspect");
        getfullargspec = PyObject_GetAttrString(inspect, "getfullargspec");
        argspec = PyObject_CallFunctionObjArgs(getfullargspec, pFunc, NULL);
        args = PyTuple_GetItem(argspec, 0);
        int isArgs = PyObject_RichCompareBool(args, Py_None, Py_EQ);
        post("isArgs: %d", isArgs);

        int py_args = PyObject_Size(args);
        // check if function if not *args or **kwargs
        if (args == Py_None){
            x->py_arg_numbers = -1; 
            post("[py4pd] The '%s' function has *args or **kwargs!", function_name->s_name);        
        }
        else {
            x->py_arg_numbers = py_args;
            post("[py4pd] The '%s' function has %d arguments!", function_name->s_name, py_args);
        }        
        x->function = pFunc;
        x->module = pModule;
        x->script_name = script_file_name;
        x->function_name = function_name; 
        x->function_called = 1;

    } else {
        // post PyErr_Print() in pd
        pd_error(x, "[py4pd] Function %s not loaded!", function_name->s_name);
        x->function_called = 0; // set the flag to 0 because it crash Pd if user try to use args method
        x->function_name = NULL;
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Set function had failed:\n %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(pModule);
        Py_XDECREF(pFunc);
        Py_XDECREF(pName);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
    }
    return;
}

// ============================================

static void runList_function(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    int listStarted = 0;
    int start = 0;
    int end = 0;

    PyObject *pArgs, *pValue;
    pArgs = PyTuple_New(2);
    t_atom *argsInsideList = (t_atom *)getbytes(argc * sizeof(t_atom));
    int argsInsideListCounter = 0;
    int pyArgs = 0;
    // pFunc = x->function;
    for (int j = 0; j < argc; ++j){
        if (argv[j].a_type == A_SYMBOL || listStarted == 1){ // loop for argc
            int k;
            if (argv[j].a_type == A_SYMBOL && argv[j].a_w.w_symbol->s_name[0] == '['){
                char *str = (char *)argv[j].a_w.w_symbol->s_name;
                str++;
                start = j;
                t_atom strAtom;
                SETSYMBOL(&strAtom, gensym(str));
                argsInsideList[argsInsideListCounter] = strAtom;
                argsInsideListCounter++;
                listStarted = 1;
            }
            else if (argv[j].a_type == A_FLOAT && listStarted == 1){
                argsInsideList[argsInsideListCounter] = argv[j];
                argsInsideListCounter++;

            }
            else if (argv[j].a_type == A_SYMBOL && argv[j].a_w.w_symbol->s_name[0] != '['){
                int lenSymbol = 0;
                for (k = 0; k < (int)strlen(argv[j].a_w.w_symbol->s_name); ++k){
                    if (argv[j].a_w.w_symbol->s_name[k] == ']'){
                        lenSymbol = k;
                        char *str = (char *)argv[j].a_w.w_symbol->s_name;
                        str[lenSymbol] = '\0';
                        end = j;
                        // convert str to atom
                        t_atom strAtom;
                        SETSYMBOL(&strAtom, gensym(str));
                        argsInsideList[argsInsideListCounter] = strAtom;
                        argsInsideListCounter++;
                        listStarted = 0;
                        PyObject *C2Python = PyList_New(0);
                        for (k = start; k <= end; ++k){
                            if (argsInsideList[k].a_type == A_SYMBOL){
                                pValue = PyUnicode_FromString(argsInsideList[k].a_w.w_symbol->s_name);
                                PyList_Append(C2Python, pValue);
                            }
                            else{
                                pValue = PyFloat_FromDouble(argsInsideList[k].a_w.w_float);
                                PyList_Append(C2Python, pValue);
                            }
                        }
                        // add the list to the args
                        PyTuple_SetItem(pArgs, pyArgs, C2Python);
                        pyArgs++;
                        argsInsideListCounter = 0;
                        break;
                    }
                }
                if (end == 0 && argv[j].a_type == A_SYMBOL){
                    argsInsideList[argsInsideListCounter] = argv[j];
                    argsInsideListCounter++;
                }
            }
            else{
                argsInsideList[argsInsideListCounter] = argv[j];
                argsInsideListCounter++;
            }
        }
        else {
            t_atom *argv_i = malloc(sizeof(t_atom)); // TODO: Check if this is necessary
            *argv_i = argv[j];
            pValue = py4pd_convert_to_python(argv_i);
            if (!pValue) {
                pd_error(x, "[py4pd] Cannot convert argument\n"); 
                return;
            }
            PyTuple_SetItem(pArgs, pyArgs, pValue);
            pyArgs++;
            // post('here');
        }
    }

    // call the function
    post("Size: %d", PyTuple_GET_SIZE(pArgs));
    post("ok");
    return;
}

// ============================================

static void run_function(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;

    if (argc != x->py_arg_numbers) {
        // check if some t_atom is an Symbol
        pd_error(x, "[py4pd] Wrong number of arguments! The function %s needs %i arguments, received %i!", x->function_name->s_name, (int)x->py_arg_numbers, argc);
        return;
    }
    
    if (x->function_called == 0) {
        pd_error(x, "[py4pd] The function %s was not called!", x->function_name->s_name);
        return;
    }

    PyObject *pFunc, *pArgs, *pValue; // pDict, *pModule,
    pFunc = x->function; // this makes the function callable 
    pArgs = PyTuple_New(argc);
    // DOC: CONVERTION TO PYTHON OBJECTS
    int i;
    for (i = 0; i < argc; ++i) {
        t_atom *argv_i = malloc(sizeof(t_atom)); // TODO: Check if this is necessary
        *argv_i = argv[i];
        pValue = py4pd_convert_to_python(argv_i);
        if (!pValue) {
            pd_error(x, "[py4pd] Cannot convert argument\n"); 
            return;
        }
        PyTuple_SetItem(pArgs, i, pValue); // DOC: Set the argument in the tuple
    }

    pValue = PyObject_CallObject(pFunc, pArgs);
    if (pValue != NULL) {                                // DOC: if the function returns a value   
        py4pd_convert_to_pd(x, pValue); // DOC: convert the value to pd        
    }
    else { // DOC: if the function returns a error
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
    }
    Py_DECREF(pArgs);
    return;
}

// ============================================

static void run_function_thread(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    
    if (argc != x->py_arg_numbers) {
        pd_error(x, "[py4pd] Wrong number of arguments!");
        return;
    }
    PyObject *pFunc, *pArgs, *pValue; // pDict, *pModule,
    pFunc = x->function;
    pArgs = PyTuple_New(argc);
    int i;
    if (x->function_called == 0) { // if the set method was not called, then we can not run the function :)
        if(pFunc != NULL){
            // create t_atom *argv from x->script_name and x->function_name
            t_atom *newargv = malloc(sizeof(t_atom) * 2);
            SETSYMBOL(newargv, x->script_name);
            SETSYMBOL(newargv+1, x->function_name);
            set_function(x, NULL, 2, newargv);
        } else{
            pd_error(x, "[py4pd] The message need to be formatted like 'set {script_name} {function_name}'!");
            return;
        }
    }
    
    // DOC: CONVERTION TO PYTHON OBJECTS
    // create an array of t_atom to store the list
    // t_atom *list = malloc(sizeof(t_atom) * argc);
    for (i = 0; i < argc; ++i) {
        t_atom *argv_i = malloc(sizeof(t_atom));
        *argv_i = argv[i];
        pValue = py4pd_convert_to_python(argv_i);
        if (!pValue) {
            pd_error(x, "[py4pd] Cannot convert argument\n");
            return;
        }
        PyTuple_SetItem(pArgs, i, pValue); // DOC: Set the argument in the tuple
    }

    pValue = PyObject_CallObject(pFunc, pArgs); // DOC: Call and execute the function
    if (pValue != NULL) {                                // DOC: if the function returns a value   
        // convert the python object to a t_atom
        py4pd_convert_to_pd(x, pValue);
    }
    else { // DOC: if the function returns a error
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_DECREF(pvalue);
        Py_DECREF(ptype);
        Py_DECREF(ptraceback);
    }
    return;
}

// // ============================================
//
// struct thread_arg_struct {
//     t_py x;
//     t_symbol s;
//     int argc;
//     t_atom *argv;
//     PyThreadState *interp;
// } thread_arg;
//
// // ============================================
//
// static void *ThreadFunc(void *lpParameter) {
//     struct thread_arg_struct *arg = (struct thread_arg_struct *)lpParameter;
//     t_py *x = &arg->x; 
//     t_symbol *s = &arg->s;
//     int argc = arg->argc;
//     t_atom *argv = arg->argv;
//     int object_number = x->object_number;
//     thread_status[object_number] = 1;
//     // PyGILState_STATE gstate;
//     running_some_thread = 1;
//     run_function_thread(x, s, argc, argv);  
//     thread_status[object_number] = 0;
//     running_some_thread = 0;
//     return 0;
// }
//
// // ============================================
//
// static void create_thread(t_py *x, t_symbol *s, int argc, t_atom *argv){
//     (void)s;
//     struct thread_arg_struct *arg = (struct thread_arg_struct *)malloc(sizeof(struct thread_arg_struct));
//     arg->x = *x;
//     arg->argc = argc;
//     arg->argv = argv;
//     int object_number = x->object_number;
//     if (x->function_called == 0) {
//         // Pd is crashing when I try to create a thread.
//         pd_error(x, "[py4pd] You need to call a function before run!");
//         free(arg);
//         return;
//     } else {
//         if (thread_status[object_number] == 0){
//             // PyThread is not thread safe, so we need to lock the GIL
//             pthread_t thread;
//             pthread_create(&thread, NULL, ThreadFunc, arg);
//             x->state = 1;
//             // check the Thread was created
//         } else {
//             pd_error(x, "[py4pd] There is a thread running in this Object!");
//             free(arg);
//         }
//     }
//     return;
// }

// ============================================
static void run(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    // check if function was called
    if (x->function_called == 0) {
        pd_error(x, "[py4pd] You need to call a function before run!");
        return;
    }
    int thread = x->thread;
    if (thread == 1) {
        run_function(x, s, argc, argv);
        pd_error(x, "[py4pd] Not implemenented! Wait for approval of PEP 684");
    } else if (thread == 2) {
        // declare state
        run_function(x, s, argc, argv);
        
    } else {
        pd_error(x, "[py4pd] Thread not created");
    }
    return;
}

// ============================================
static void restartPython(t_py *x){
    Py_Finalize();
    post("[py4pd] Python interpreter was restarted!");
    x->function_called = 0;
    x->function_name = NULL;
    x->script_name = NULL;
    x->py_main_interpreter = NULL;
    x->module = NULL;
    x->function = NULL;
    int i;
    for (i = 0; i < 100; i++) {
        t_py *y = py4pd_object_array[i];
        if (y != NULL) {
            y->function_called = 0;
            y->function_name = NULL;
            y->script_name = NULL;
            y->py_main_interpreter = NULL;
            y->module = NULL;
            y->function = NULL;
        }
    }
    Py_Initialize();
    return;
}

// ============================================
// ============================================
// ============================================

static void thread(t_py *x, t_floatarg f){
    int thread = (int)f;
    if (thread == 1) {
        post("[py4pd] Threading enabled, but not working yet!");
        x->thread = 1;
        return;
    } else if (thread == 0) {
        x->thread = 2; // 
        post("[py4pd] Threading disabled");
        return;
    } else {
        pd_error(x, "[py4pd] Threading status must be 0 or 1");
    }
}

// ============================================
// =========== SETUP OF OBJECT ================
// ============================================

void *py4pd_new(t_symbol *s, int argc, t_atom *argv){ 
    if (!Py_IsInitialized()) {
        // Credits
        post("");
        post("[py4pd] by Charles K. Neimog");
        post("[py4pd] Version 0.5.0       ");
        post("[py4pd] Python version %d.%d.%d", PY_MAJOR_VERSION, PY_MINOR_VERSION, PY_MICRO_VERSION);
        post("[py4pd] Inspired by the work of Thomas Grill and SOPI research group.");
        post("");
        PyImport_AppendInittab("pd", PyInit_pd); // Add the pd module to the python interpreter
        Py_InitializeEx(1); // DOC: Initialize the Python interpreter. If 1, the signal handler is installed.
    }
    object_count++; // count the number of objects
    t_py *x = (t_py *)pd_new(py4pd_class); // create a new object
    x->object_number = object_count; // save object number
    x->out_A = outlet_new(&x->x_obj, 0); // cria um outlet 
    x->x_canvas = canvas_getcurrent(); // pega o canvas atual
    t_canvas *c = x->x_canvas;  // get the current canvas
    t_symbol *patch_dir = canvas_getdir(c); // directory of opened patch
    x->home_path = patch_dir;     // set name of the home path
    x->packages_path = patch_dir; // set name of the packages path
    x->thread = 2; // default is 2 (no threading)
    py4pd_object_array[object_count] = x; // save the object in the array
    set_py4pd_config(x); // set the config file
    if (argc > 1) { // check if there are two arguments
        set_function(x, s, argc, argv); // this not work with python submodules
    }
    return(x);
}

// ============================================

void py4pd_free(t_py *x){
    PyObject  *pModule, *pFunc; // pDict, *pName,
    pFunc = x->function;
    pModule = x->module;
    object_count--;
    // clear all struct
    if (pModule != NULL) {
        Py_DECREF(pModule);
    }
    if (pFunc != NULL) {
        Py_DECREF(pFunc);
    }
    if (object_count == 1) {
        Py_Finalize();
        object_count = 0;
        post("[py4pd] Python interpreter finalized");
    }
}

// ====================================================

void py4pd_setup(void){
    py4pd_class =       class_new(gensym("py4pd"), // cria o objeto quando escrevemos py4pd
                        (t_newmethod)py4pd_new, // metodo de criação do objeto             
                        (t_method)py4pd_free, // quando voce deleta o objeto
                        sizeof(t_py), // quanta memoria precisamos para esse objeto
                        CLASS_DEFAULT, // nao há uma GUI especial para esse objeto???
                        A_GIMME, // o argumento é um símbolo
                        0); // todos os outros argumentos por exemplo um numero seria A_DEFFLOAT
    
    // Config
    class_addmethod(py4pd_class, (t_method)home, gensym("home"), A_GIMME, 0); // set home path
    class_addmethod(py4pd_class, (t_method)packages, gensym("packages"), A_GIMME, 0); // set packages path
    class_addmethod(py4pd_class, (t_method)thread, gensym("thread"), A_FLOAT, 0); // on/off threading
    class_addmethod(py4pd_class, (t_method)reload, gensym("reload"), 0, 0); // reload python script
    class_addmethod(py4pd_class, (t_method)restartPython, gensym("restart"), 0, 0); // it restart python interpreter

    // Edit py code
    class_addmethod(py4pd_class, (t_method)vscode, gensym("vscode"), 0, 0); // open vscode
    class_addmethod(py4pd_class, (t_method)create, gensym("create"), A_GIMME, 0); // create file or open it
    class_addmethod(py4pd_class, (t_method)vscode, gensym("click"), 0, 0); // when click open vscode
    
    // User use
    class_addmethod(py4pd_class, (t_method)documentation, gensym("doc"), 0, 0); // open documentation
    class_addmethod(py4pd_class, (t_method)run, gensym("run"), A_GIMME, 0);  // run function
    class_addmethod(py4pd_class, (t_method)runList_function, gensym("runlist"), A_GIMME, 0);  // run function TODO:
    class_addmethod(py4pd_class, (t_method)set_function, gensym("set"), A_GIMME, 0); // set function to be called




}


// // dll export function
#ifdef _WIN64

__declspec(dllexport) void py4pd_setup(void); // when I add python module, for some reson, pd not see py4pd_setup

#endif

