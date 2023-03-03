#include "py4pd.h"
#include "m_pd.h"
#include "pd_module.h"
#include "py4pd_utils.h"
#include "py4pd_pic.h"
#include <math.h>

// Include numpy
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// t_py *py4pd_object_array[100];
t_class *py4pd_class; // DOC: For audio in and without audio
t_class *py4pd_class_VIS; // DOC: For visualisation | pic object by pd-else
t_class *py4pd_classAudioOut; // DOC: For audio out
t_class *edit_proxy_class;

int object_count;

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
                post("");
                post("==== %s documentation ====", x->function_name->s_name);
                post("");
                post("%s", Doc);
                post("");
                post("==== %s documentation ====", x->function_name->s_name);
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
void pd4py_system_func (const char *command){
    int result = system(command);
    if (result == -1){
        post("[py4pd] %s", command);
        return;
    }
}

// ============================================
static void open(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    (void)argc;

    if (argv[0].a_type != A_SYMBOL) {
        pd_error(x, "[py4pd] The script name must be a symbol");
        return;
    }

    x->script_name = argv[0].a_w.w_symbol;
    
    // Open VsCode in Windows
    #ifdef _WIN64 
    char *command = get_editor_command(x);
    command = get_editor_command(x);
    // use get_editor_command
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
    command = get_editor_command(x);
    pd4py_system_func(command);
    #endif

    // If macOS
    #ifdef __APPLE__
    pd_error(x, "Not tested in your Platform, please send me a report!");
    #endif
    return ;

}

// ====================================
static void editor(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    if (argc != 0){
        x->editorName = atom_getsymbol(argv+0);
        post("[py4pd] Editor set to: %s", x->editorName->s_name);
        return;
    }

    if (x->function_called == 0) { // if the set method was not called, then we can not run the function :)
        pd_error(x, "[py4pd] To open vscode you need to set the function first!");
        return;
    }
    post("[py4pd] Opening editor...");

    // Open VsCode in Windows
    #ifdef _WIN64 
    char *command = get_editor_command(x);
    command = get_editor_command(x);
    // use get_editor_command
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
    command = get_editor_command(x);
    pd4py_system_func(command);
    #endif

    // If macOS
    #ifdef __APPLE__
    pd_error(x, "Not tested in your Platform, please send me a report!");
    #endif
    return ;

}

// ====================================
static void vscode(t_py *x){
    pd_error(x, "This method is deprecated, please use the editor method instead!");
    editor(x, NULL, 0, NULL);
}

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
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        return;
    } 
    else{
        Py_XDECREF(x->module);
        pFunc = PyObject_GetAttrString(pModule, x->function_name->s_name); // Function name inside the script file
        Py_ssize_t refcount = Py_REFCNT(pName);
        post("Reference cont pName: %i", refcount);
        refcount = Py_REFCNT(pReload);
        post("Reference cont pReload: %i", refcount);
        Py_DECREF(pName);
        Py_DECREF(pReload);
        if (pFunc && PyCallable_Check(pFunc)){ // Check if the function exists and is callable 
            x->function = pFunc;
            x->module = pModule;
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
            x->function_called = 0;
        }
    }
    
    // Check if there is extension (not to use it)
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
    // check number of arguments
    if (argc < 2) { // check is the number of arguments is correct | set "function_script" "function_name"
        pd_error(x,"[py4pd] 'set' message needs two arguments! The 'Script Name' and the 'Function Name'!");
        return;
    }
    // =====================
    PyObject *pName, *pModule, *pFunc; // Create the variables of the python objects
    
    // =====================
    // Add aditional path to python to work with Pure Data
    PyObject *home_path = PyUnicode_FromString(x->home_path->s_name); // Place where script file will probably be
    PyObject *site_package = PyUnicode_FromString(x->packages_path->s_name); // Place where the packages will be
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
    Py_DECREF(pName); // Delete the name of the script file
    if (pFunc && PyCallable_Check(pFunc)){ // Check if the function exists and is callable   
        PyObject *inspect=NULL, *getfullargspec=NULL, *argspec=NULL, *args=NULL;
        inspect = PyImport_ImportModule("inspect");
        getfullargspec = PyObject_GetAttrString(inspect, "getfullargspec");
        argspec = PyObject_CallFunctionObjArgs(getfullargspec, pFunc, NULL);
        args = PyTuple_GetItem(argspec, 0);
        // int isArgs = PyObject_RichCompareBool(args, Py_None, Py_EQ); // TODO: way to check if function has *args or **kwargs
        int py_args = PyObject_Size(args);
        if (args == Py_None){
            x->py_arg_numbers = -1; 
            post("[py4pd] The '%s' function has *args or **kwargs!", function_name->s_name);        
        }
        else {
            x->py_arg_numbers = py_args;
            post("[py4pd] The '%s' function has %d arguments!", function_name->s_name, py_args);
        }        
        Py_DECREF(inspect);
        Py_DECREF(getfullargspec);
        Py_DECREF(argspec);
        Py_DECREF(args);
        Py_DECREF(pModule);

        x->function = pFunc;
        x->script_name = script_file_name;
        x->function_name = function_name; 
        x->function_called = 1;

    } else {
        pd_error(x, "[py4pd] Function %s not loaded!", function_name->s_name);
        x->function_called = 1; // set the flag to 0 because it crash Pd if user try to use args method
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Set function had failed: %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        PyErr_Clear();

    }
    return;
}

// ============================================
static void run_function(t_py *x, t_symbol *s, int argc, t_atom *argv){
    //  TODO: Check for memory leaks 
    (void)s;
    int OpenList_count = 0;
    int CloseList_count = 0;

    PyObject *pValue, *ArgsTuple; 
    if (argc != 0){
        for (int i = 0; i < argc; i++) {
            if (argv[i].a_type == A_SYMBOL){
                if (strchr(argv[i].a_w.w_symbol->s_name, '[') != NULL){
                    CloseList_count++;
                }
                if (strchr(argv[i].a_w.w_symbol->s_name, ']') != NULL){
                    OpenList_count++;
                }
            }
        }
        if (OpenList_count != CloseList_count){
            pd_error(x, "[py4pd] The number of '[' and ']' is not the same!");
            return;
        }
        PyObject *lists[OpenList_count]; // create a list of lists 
        
        ArgsTuple = py4pd_convert_to_py(lists, argc, argv); // convert the arguments to python
        int argCount = PyTuple_Size(ArgsTuple); // get the number of arguments
        if (argCount != x->py_arg_numbers) {
            pd_error(x, "[py4pd] Wrong number of arguments! The function %s needs %i arguments, received %i!", x->function_name->s_name, (int)x->py_arg_numbers, argCount);
            post("Length of tuple: %i", argCount);
            post("Length of args: %i", x->py_arg_numbers);
            return;
        }
    }
    else {
        ArgsTuple = PyTuple_New(0);
    }


    // WARNING: this can generate errors? How this will work on multithreading?
    PyObject *capsule = PyCapsule_New(x, "py4pd", NULL); // create a capsule to pass the object to the python interpreter
    PyModule_AddObject(PyImport_AddModule("__main__"), "py4pd", capsule); // add the capsule to the python interpreter
    pValue = PyObject_CallObject(x->function, ArgsTuple);
    if (pValue != NULL) {                                // if the function returns a value   
        py4pd_convert_to_pd(x, pValue); // convert the value to pd        
    }
    else { // if the function returns a error
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
    // free lists
    Py_XDECREF(pValue);
    Py_DECREF(ArgsTuple);
    return;
}

// ============================================
/*
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
   
    // CONVERTION TO PYTHON OBJECTS
    // create an array of t_atom to store the list
    t_atom *list = malloc(sizeof(t_atom) * argc);
    for (i = 0; i < argc; ++i) {
        t_atom *argv_i = malloc(sizeof(t_atom));
        *argv_i = argv[i];
        pValue = py4pd_convert_to_python(argv_i);
        if (!pValue) {
            pd_error(x, "[py4pd] Cannot convert argument\n");
            return;
        }
        PyTuple_SetItem(pArgs, i, pValue); // Set the argument in the tuple
    }

    pValue = PyObject_CallObject(pFunc, pArgs); // Call and execute the function
    if (pValue != NULL) {                                // if the function returns a value   
        // convert the python object to a t_atom
        py4pd_convert_to_pd(x, pValue);
    }
    else { // if the function returns a error
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

// ============================================

struct thread_arg_struct {
    t_py x;
    t_symbol s;
    int argc;
    t_atom *argv;
    PyThreadState *interp;
} thread_arg;

// // ============================================

static void *ThreadFunc(void *lpParameter) {
    struct thread_arg_struct *arg = (struct thread_arg_struct *)lpParameter;
    t_py *x = &arg->x; 
    t_symbol *s = &arg->s;
    int argc = arg->argc;
    t_atom *argv = arg->argv;
    int object_number = x->object_number;
    thread_status[object_number] = 1;
    // PyGILState_STATE gstate;
    running_some_thread = 1;
    run_function_thread(x, s, argc, argv);  
    thread_status[object_number] = 0;
    running_some_thread = 0;
    return 0;
}

// ============================================

static void create_thread(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    struct thread_arg_struct *arg = (struct thread_arg_struct *)malloc(sizeof(struct thread_arg_struct));
    arg->x = *x;
    arg->argc = argc;
    arg->argv = argv;
    int object_number = x->object_number;
    if (x->function_called == 0) {
        // Pd is crashing when I try to create a thread.
        pd_error(x, "[py4pd] You need to call a function before run!");
        free(arg);
        return;
    } else {
        if (thread_status[object_number] == 0){
            // PyThread is not thread safe, so we need to lock the GIL
            pthread_t thread;
            pthread_create(&thread, NULL, ThreadFunc, arg);
            x->state = 1;
            // check the Thread was created
        } else {
            pd_error(x, "[py4pd] There is a thread running in this Object!");
            free(arg);
        }
    }
    return;
}
*/

// ============================================
static void run(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    if (x->function_called == 0) {
        pd_error(x, "[py4pd] You need to call a function before run!");
        return;
    }
    if (x->thread == 1) {
        run_function(x, s, argc, argv);
        pd_error(x, "[py4pd] Not implemenented! Wait for approval of PEP 684");
    } else if (x->thread == 0) {
        run_function(x, s, argc, argv);
        
    } else {
        pd_error(x, "[py4pd] Thread not created");
    }
    return;
}

// ============================================
t_int *py4pd_perform(t_int *w){

    t_py *x = (t_py *)(w[1]); // this is the object itself
    if (x->audioInput == 0 && x->audioOutput == 0) {
        return (w + 4);
    }    
    t_sample *audioIn = (t_sample *)(w[2]); // this is the input vector (the sound)
    int n = (int)(w[3]); // this is the vector size (number of samples, for example 64)
    const npy_intp dims = n;
    PyObject *ArgsTuple, *pValue, *pAudio, *pSample;

    if (x->function_called == 0) {
        pd_error(x, "[py4pd] You need to call a function before run!");
        return (w + 4);
    }

    pSample = NULL; //  
    if (x->use_NumpyArray == 1) {
        pAudio = PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT, audioIn);
        ArgsTuple = PyTuple_New(1);
        PyTuple_SetItem(ArgsTuple, 0, pAudio);
    }
    else {
        pAudio = PyList_New(n);
        for (int i = 0; i < n; i++) {
            pSample = PyFloat_FromDouble(audioIn[i]);
            PyList_SetItem(pAudio, i, pSample);
        }
        ArgsTuple = PyTuple_New(1);
        PyTuple_SetItem(ArgsTuple, 0, pAudio);
    }

    // WARNING: this can generate errors? How this will work on multithreading?
    PyObject *capsule = PyCapsule_New(x, "py4pd", NULL); // create a capsule to pass the object to the python interpreter
    PyModule_AddObject(PyImport_AddModule("__main__"), "py4pd", capsule); // add the capsule to the python interpreter

    // call the function
    pValue = PyObject_CallObject(x->function, ArgsTuple);
    if (pValue != NULL) {                               
        py4pd_convert_to_pd(x, pValue); // convert the value to pd        
    }
    else { // if the function returns a error
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
    Py_DECREF(ArgsTuple);
    if (pSample != NULL) { 
        Py_DECREF(pSample);
    }
    return (w + 4);
}

// ============================================
t_int *py4pd_performAudioOutput(t_int *w){
    //  TODO: Check for memory leaks

    t_py *x = (t_py *)(w[1]); // this is the object itself
    if (x->audioInput == 0 && x->audioOutput == 0) {
        return (w + 5);
    }
    if (x->function_called == 0) {
        pd_error(x, "[py4pd] You need to call a function before run!");
        return (w + 5);
    }
    t_sample *audioIn = (t_sample *)(w[2]); // this is the input vector (the sound)
    t_sample *audioOut = (t_sample *)(w[3]); // this is the output vector (the sound)
    int n = (int)(w[4]); // this is the vector size (number of samples, for example 64)
    const npy_intp dims = n;
    PyObject *ArgsTuple, *pValue, *pAudio, *pSample;
    pSample = NULL;   // NOTE: This is the way to not distorce the audio output
    if (x->use_NumpyArray == 1) {
        pAudio = PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT, audioIn);
        ArgsTuple = PyTuple_New(1);
        PyTuple_SetItem(ArgsTuple, 0, pAudio);
    }
    else {
        // pSample = NULL;  NOTE: this change the sound.
        // pAudio = PyList_New(n); cconvert audioIn in tuple
        pAudio = PyTuple_New(n);
        for (int i = 0; i < n; i++) {
            pSample = PyFloat_FromDouble(audioIn[i]);
            PyTuple_SetItem(pAudio, i, pSample);
        }
        ArgsTuple = PyTuple_New(1);
        PyTuple_SetItem(ArgsTuple, 0, pAudio);
        // if (pSample != NULL) { NOTE: this change the sound.
        //     Py_DECREF(pSample);
        // }
    }
    // WARNING: this can generate errors? How this will work on multithreading? || In PEP 684 this will be per interpreter or global?
    PyObject *capsule = PyCapsule_New(x, "py4pd", NULL); // create a capsule to pass the object to the python interpreter
    PyModule_AddObject(PyImport_AddModule("__main__"), "py4pd", capsule); // add the capsule to the python interpreter
    pValue = PyObject_CallObject(x->function, ArgsTuple);

    if (pValue != NULL) {                               
        if (PyList_Check(pValue)){
            for (int i = 0; i < n; i++) {
                audioOut[i] = PyFloat_AsDouble(PyList_GetItem(pValue, i));
            }          
        }
        else if (PyTuple_Check(pValue)){
            for (int i = 0; i < n; i++) {
                audioOut[i] = PyFloat_AsDouble(PyTuple_GetItem(pValue, i));
            }          
        }
        else if (x->numpyImported == 1) {
            if (PyArray_Check(pValue)) {
                PyArrayObject *pArray = (PyArrayObject *)pValue;
                for (int i = 0; i < n; i++) { // TODO: try to add audio support without another loop
                    audioOut[i] = PyFloat_AsDouble(PyArray_GETITEM(pArray, PyArray_GETPTR1(pArray, i)));
                }
            }
            else{
                pd_error(x, "[py4pd] The function must return a list, a tuple or a numpy array, returned: %s", pValue->ob_type->tp_name);
            }
        }
        else{
            pd_error(x, "[py4pd] The function must return a list, since numpy array is disabled, returned: %s", pValue->ob_type->tp_name);
        }
    }
    else { // if the function returns a error
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
    Py_DECREF(ArgsTuple);
    if (pSample != NULL) {
        Py_DECREF(pSample);
    }
    return (w + 5);
}

// ============================================
static void py4pd_dspin(t_py *x, t_signal **sp){
    if (x->audioOutput == 0){
        dsp_add(py4pd_perform, 3, x, sp[0]->s_vec, sp[0]->s_n);
    }
    else { // python output is audio
       dsp_add(py4pd_performAudioOutput, 4, x, sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_n);
    }
}
// ============================================
static void restartPython(t_py *x){
    pd_error(x, "[py4pd] This function is not implemented yet!");
    return;

    // Py_Finalize();
    // x->function_called = 0;
    // x->function_name = NULL;
    // x->script_name = NULL;
    // x->module = NULL;
    // x->function = NULL;
    // int i;
    // for (i = 0; i < 100; i++) {
    //     char object_name[20];
    //     sprintf(object_name, "py4pd_%d", i);
    //     post("object name: %s", object_name);
        // y = (t_py *)pd_findbyclass((x->object_name = gensym(object_name)), py4pd_class);
        // post("object pointer: %p", y); 

        // if (y != NULL) {
        //     y->function_called = 0;
        //     y->function_name = NULL;
        //     y->script_name = NULL;
        //     y->module = NULL;
        //     y->function = NULL;
        //     y->packages_path = gensym("./py-modules");
        //     y->thread = 2;
        //     y->editorName = gensym("code");
        // }
    // PyImport_AppendInittab("pd", PyInit_pd); // Add the pd module to the python interpreter
    // Py_Initialize();
    // return;
}

// ============================================
static void *py4pdImportNumpy(){
    import_array();
    return NULL;
}

// ============================================
static void usenumpy(t_py *x, t_floatarg f){
    //  TODO: If the run method set before the end of the thread, there is an error, that close all PureData.
    int usenumpy = (int)f;
    if (usenumpy == 1) {
        post("[py4pd] Numpy Array enabled.");
        x->use_NumpyArray = 1;
        if (x->numpyImported == 0) {
            py4pdImportNumpy();
            x->numpyImported = 1;
        }
    } else if (usenumpy == 0) {
        x->use_NumpyArray = 0; 
        post("[py4pd] Numpy Array disabled");
    } else {
        pd_error(x, "[py4pd] Numpy status must be 0 (disable) or 1 (enable)");
    }
    return;
}

// ===========================================

static void thread(t_py *x, t_floatarg f){

    //  TODO: If the run method set before the end of the thread, there is an error, that close all PureData.
    int thread = (int)f;
    if (thread == 1) {
        post("[py4pd] Threading enabled, wait for approval of PEP 684");
        x->thread = 1;
        return;
    } else if (thread == 0) {
        x->thread = 0; 
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
    int i;
    t_py *x;
    int visMODE = 0;
    int audioOUT = 0;
    int normalMODE = 1;

    for (i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            t_symbol *py4pdArgs = atom_getsymbolarg(i, argc, argv);
            if (py4pdArgs == gensym("-picture") || py4pdArgs == gensym("-score") || py4pdArgs == gensym("-canvas")){
                visMODE = 1;
             }
            else if (py4pdArgs == gensym("-audio") || py4pdArgs == gensym("-audioout")){
                audioOUT = 1;
            }
        }
    }
    if (visMODE == 1 && audioOUT == 0){
        x = (t_py *)pd_new(py4pd_class_VIS); // create a new object
        // post("py4pd: visual mode");
    }
    else if (audioOUT == 1 && visMODE == 0){
        x = (t_py *)pd_new(py4pd_classAudioOut); // create a new object
        // post("py4pd: audio mode");
    }
    else if (normalMODE == 1){
        x = (t_py *)pd_new(py4pd_class); // create a new object
        // post("py4pd: normal/analisys mode");
    }
    else {
        post("Error in py4pd_new, please report this error to the developer, this message should not appear.");
        return NULL;
    }

    x->x_canvas = canvas_getcurrent(); // pega o canvas atual
    t_canvas *c = x->x_canvas;  // get the current canvas
    t_symbol *patch_dir = canvas_getdir(c); // directory of opened patch
    x->audioInput = 0;
    x->audioOutput = 0;
    x->visMode = 0;

    if (!Py_IsInitialized()) {
        object_count = 1;  // To count the numbers of objects, and finalize the interpreter when the last object is deleted
        post("");
        post("[py4pd] by Charles K. Neimog");
        post("[py4pd] Version 0.6.0       ");
        post("[py4pd] Python version %d.%d.%d", PY_MAJOR_VERSION, PY_MINOR_VERSION, PY_MICRO_VERSION);
        post("");
        PyImport_AppendInittab("pd", PyInit_pd); // Add the pd module to the python interpreter
        Py_Initialize(); // Initialize the Python interpreter. If 1, the signal handler is installed.    
    }

    object_count++; // count the number of objects;  
    for (i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            t_symbol *py4pdArgs = atom_getsymbolarg(i, argc, argv);
            if (py4pdArgs == gensym("-picture") || py4pdArgs == gensym("-score") ||  py4pdArgs == gensym("-canvas")){
                py4pd_InitVisMode(x, c, py4pdArgs, i, argc, argv);
                x->visMode = 1;
                x->x_outline = 1;
            }
            else if (py4pdArgs == gensym("-audioout")) {
                // post("[py4pd] Audio Outlets enabled");
                x->audioOutput = 1;
                x->use_NumpyArray = 0;
                x->out_A = outlet_new(&x->x_obj, gensym("signal")); // create a signal outlet
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j+1];
                }
                argc--;
            }
            else if (py4pdArgs == gensym("-nvim") || py4pdArgs == gensym("-vscode") || py4pdArgs == gensym("-emacs")) {
                x->editorName = gensym(py4pdArgs->s_name + 1);
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j+1];
                }
                argc--;
            }
            else if (py4pdArgs == gensym("-audioin")) {
                x->audioInput = 1;
                x->use_NumpyArray = 0;
            }
            else if (py4pdArgs == gensym("-audio")){
                x->audioInput = 1;
                x->audioOutput = 1;
                x->out_A = outlet_new(&x->x_obj, gensym("signal")); // create a signal outlet
                x->use_NumpyArray = 0;
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j+1];
                }
                argc--;
            }
        }
    }

    if (x->audioOutput == 0){
        x->out_A = outlet_new(&x->x_obj, 0); // cria um outlet 
    }
    x->thread = 0; 
    x->object_number = object_count; // save object number
    x->home_path = patch_dir;     // set name of the home path
    x->packages_path = patch_dir; // set name of the packages path
    set_py4pd_config(x); // set the config file (in py4pd.cfg, make this be saved in the patch)
    if (argc > 1) { // check if there are two arguments
        set_function(x, s, argc, argv); 
        import_array(); // import numpy
        x->numpyImported = 1;
    }
    return(x);
}

// ============================================
void *py4pd_free(t_py *x){
    object_count--;

    if (x->visMode == 1){
        PY4PD_free(x);
    }

    if (object_count == 1) {
        // Py_Finalize(); // BUG: Not possible because it crashes if another py4pd is created in the same PureData session
    }
    return (void *)x;
}

// ====================================================
void py4pd_setup(void){
    py4pd_class =       class_new(gensym("py4pd"), // cria o objeto quando escrevemos py4pd
                        (t_newmethod)py4pd_new, // metodo de criação do objeto             
                        (t_method)py4pd_free, // quando voce deleta o objeto
                        sizeof(t_py), // quanta memoria precisamos para esse objeto
                        0, // nao há uma GUI especial para esse objeto???
                        A_GIMME, // o argumento é um símbolo
                        0); 
    py4pd_class_VIS =   class_new(gensym("py4pd"), 
                        (t_newmethod)py4pd_new, 
                        (t_method)py4pd_free, 
                        sizeof(t_py), 
                        0, 
                        A_GIMME, 
                        0); 
    py4pd_classAudioOut =   class_new(gensym("py4pd"), 
                        (t_newmethod)py4pd_new, 
                        (t_method)py4pd_free, 
                        sizeof(t_py), 
                        0, 
                        A_GIMME, 
                        0); 


    // Sound in
    class_addmethod(py4pd_class, (t_method)py4pd_dspin, gensym("dsp"), A_CANT, 0); // add a method to a class
    class_addmethod(py4pd_classAudioOut, (t_method)py4pd_dspin, gensym("dsp"), A_CANT, 0); // add a method to a class
    CLASS_MAINSIGNALIN(py4pd_class, t_py, py4pd_audio); // TODO: Repensando como fazer isso quando o áudio não for usado.
    CLASS_MAINSIGNALIN(py4pd_classAudioOut, t_py, py4pd_audio); // TODO: Repensando como fazer isso quando o áudio não for usado.
    class_addmethod(py4pd_class, (t_method)usenumpy, gensym("numpy"), A_FLOAT, 0); // add a method to a class
    class_addmethod(py4pd_classAudioOut, (t_method)usenumpy, gensym("numpy"), A_FLOAT, 0); // add a method to a class
    
    // Pic related
    class_addmethod(py4pd_class_VIS, (t_method)PY4PD_size_callback, gensym("_picsize"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(py4pd_class_VIS, (t_method)PY4PD_mouserelease, gensym("_mouserelease"), 0);
    class_addmethod(py4pd_class_VIS, (t_method)PY4PD_outline, gensym("outline"), A_DEFFLOAT, 0);
    class_addmethod(py4pd_class_VIS, (t_method)PY4PD_zoom, gensym("zoom"), A_CANT, 0);

    // this is like have lot of objects with the same name, add all methods for py4pd_class, py4pd_class_AudioOut and py4pd_class_VIS
    class_addmethod(py4pd_class, (t_method)home, gensym("home"), A_GIMME, 0); // set home path
    class_addmethod(py4pd_class_VIS, (t_method)home, gensym("home"), A_GIMME, 0); // set home path
    class_addmethod(py4pd_classAudioOut, (t_method)packages, gensym("home"), A_GIMME, 0); // set packages path

    class_addmethod(py4pd_class, (t_method)packages, gensym("packages"), A_GIMME, 0); // set packages path
    class_addmethod(py4pd_class_VIS, (t_method)packages, gensym("packages"), A_GIMME, 0); // set packages path
    class_addmethod(py4pd_classAudioOut, (t_method)packages, gensym("packages"), A_GIMME, 0); // set packages path

    class_addmethod(py4pd_class, (t_method)thread, gensym("thread"), A_FLOAT, 0); // on/off threading
    class_addmethod(py4pd_class_VIS, (t_method)thread, gensym("thread"), A_FLOAT, 0); // on/off threading
    class_addmethod(py4pd_classAudioOut, (t_method)thread, gensym("thread"), A_FLOAT, 0); // on/off threading

    class_addmethod(py4pd_class, (t_method)reload, gensym("reload"), 0, 0); // reload python script
    class_addmethod(py4pd_class_VIS, (t_method)reload, gensym("reload"), 0, 0); // reload python script
    class_addmethod(py4pd_classAudioOut, (t_method)reload, gensym("reload"), 0, 0); // reload python script

    class_addmethod(py4pd_class, (t_method)restartPython, gensym("restart"), 0, 0); // it restart python interpreter
    class_addmethod(py4pd_class_VIS, (t_method)restartPython, gensym("restart"), 0, 0); // it restart python interpreter
    class_addmethod(py4pd_classAudioOut, (t_method)restartPython, gensym("restart"), 0, 0); // it restart python interpreter

    // Edit Python Code
    class_addmethod(py4pd_class, (t_method)vscode, gensym("vscode"), 0, 0); // open editor  WARNING: will be removed
    
    class_addmethod(py4pd_class, (t_method)editor, gensym("editor"), A_GIMME, 0); // open code
    class_addmethod(py4pd_class_VIS, (t_method)editor, gensym("editor"), A_GIMME, 0); // open code
    class_addmethod(py4pd_classAudioOut, (t_method)editor, gensym("editor"), A_GIMME, 0); // open code

    class_addmethod(py4pd_class, (t_method)open, gensym("open"), A_GIMME, 0); // create file or open it TODO: fix this
    class_addmethod(py4pd_class_VIS, (t_method)open, gensym("open"), A_GIMME, 0); // create file or open it
    class_addmethod(py4pd_classAudioOut, (t_method)open, gensym("open"), A_GIMME, 0); // create file or open it

    class_addmethod(py4pd_class, (t_method)editor, gensym("click"), 0, 0); // when click open editor
    class_addmethod(py4pd_classAudioOut, (t_method)editor, gensym("click"), 0, 0); // when click open editor
    
    // User Interface
    class_addmethod(py4pd_class, (t_method)documentation, gensym("doc"), 0, 0); // open documentation
    class_addmethod(py4pd_class_VIS, (t_method)documentation, gensym("doc"), 0, 0); // open documentation
    class_addmethod(py4pd_classAudioOut, (t_method)documentation, gensym("doc"), 0, 0); // open documentation

    class_addmethod(py4pd_class, (t_method)run, gensym("run"), A_GIMME, 0);  // run function
    class_addmethod(py4pd_class_VIS, (t_method)run, gensym("run"), A_GIMME, 0);  // run function
    class_addmethod(py4pd_classAudioOut, (t_method)run, gensym("run"), A_GIMME, 0);  // run function

    class_addmethod(py4pd_class, (t_method)set_function, gensym("set"), A_GIMME,  0); // set function to be called
    class_addmethod(py4pd_class_VIS, (t_method)set_function, gensym("set"), A_GIMME,  0); // set function to be called
    class_addmethod(py4pd_classAudioOut, (t_method)set_function, gensym("set"), A_GIMME,  0); // set function to be called
    
    //  TODO: Way to set global variables, I think that will be important for things like general path;
    //  TODO: Set some audio parameters to work with py4pd_dspin, 'dspparams', 'dspparams'
    //
}

#ifdef _WIN64
__declspec(dllexport) void py4pd_setup(void); // when I add python module, for some reson, pd not see py4pd_setup
#endif


times in msec
 clock   self+sourced   self:  sourced script
 clock   elapsed:              other lines

000.028  000.028: --- NVIM STARTING ---
000.176  000.148: event init
000.272  000.097: early init
000.325  000.053: locale set
000.356  000.031: init first window
000.569  000.213: inits 1
000.580  000.010: window checked
000.582  000.002: parsing arguments
000.915  000.056  000.056: require('vim.shared')
001.049  000.061  000.061: require('vim._meta')
001.052  000.134  000.073: require('vim._editor')
001.053  000.215  000.024: require('vim._init_packages')
001.055  000.258: init lua interpreter
001.125  000.071: expanding arguments
001.163  000.038: inits 2
001.449  000.286: init highlight
001.450  000.001: waiting for UI
002.365  000.915: done waiting for UI
002.377  000.012: init screen for UI
002.509  000.132: init default mappings
002.522  000.013: init default autocommands
003.087  000.059  000.059: sourcing /usr/share/nvim/runtime/ftplugin.vim
003.169  000.027  000.027: sourcing /usr/share/nvim/runtime/indent.vim
003.213  000.021  000.021: sourcing $VIM/sysinit.vim
004.506  000.660  000.660: require('lazy')
004.542  000.019  000.019: require('ffi')
004.902  000.358  000.358: require('lazy.core.cache')
005.080  000.165  000.165: require('lazy.stats')
005.183  000.080  000.080: require('lazy.core.util')
005.336  000.150  000.150: require('lazy.core.config')
005.439  000.037  000.037: require('lazy.core.handler')
005.493  000.052  000.052: require('lazy.core.plugin')
005.499  000.161  000.072: require('lazy.core.loader')
006.637  000.091  000.091: require('lazy.core.handler.cmd')
006.693  000.050  000.050: require('lazy.core.handler.keys')
006.759  000.062  000.062: require('lazy.core.handler.event')
006.808  000.045  000.045: require('lazy.core.handler.ft')
007.277  000.324  000.324: sourcing /usr/share/nvim/runtime/filetype.lua
007.539  000.029  000.029: sourcing /home/neimog/.local/share/nvim/lazy/plenary.nvim/plugin/plenary.vim
007.746  000.043  000.043: sourcing /home/neimog/.local/share/nvim/lazy/nvim-web-devicons/plugin/nvim-web-devicons.vim
008.105  000.161  000.161: sourcing /home/neimog/.local/share/nvim/lazy/neo-tree.nvim/plugin/neo-tree.vim
008.343  000.018  000.018: sourcing /home/neimog/.local/share/nvim/lazy/py_lsp.nvim/plugin/py_lsp.vim
009.440  000.379  000.379: sourcing /home/neimog/.local/share/nvim/lazy/copilot.vim/autoload/copilot.vim
009.471  000.768  000.389: sourcing /home/neimog/.local/share/nvim/lazy/copilot.vim/plugin/copilot.vim
011.238  000.943  000.943: require('vim.filetype')
011.249  000.005  000.005: require('vim.keymap')
011.810  000.045  000.045: require('luasnip.util.types')
011.862  000.049  000.049: require('luasnip.util.ext_opts')
012.138  000.063  000.063: require('luasnip.session')
012.148  000.237  000.174: require('luasnip.util.util')
012.151  000.287  000.050: require('luasnip.extras.filetype_functions')
012.283  000.756  000.376: require('luasnip.config')
012.472  000.099  000.099: require('luasnip.session.snippet_collection')
013.015  000.496  000.496: require('luasnip.util._builtin_vars')
013.144  000.121  000.121: require('vim.inspect')
013.175  000.700  000.083: require('luasnip.util.environ')
013.252  000.076  000.076: require('luasnip.util.extend_decorator')
013.414  000.069  000.069: require('luasnip.loaders._caches')
013.481  000.065  000.065: require('luasnip.util.path')
013.493  000.238  000.104: require('luasnip.loaders')
013.543  000.046  000.046: require('luasnip.util.log')
013.555  001.266  000.107: require('luasnip')
013.589  003.406  000.435: sourcing /home/neimog/.local/share/nvim/lazy/LuaSnip/plugin/luasnip.lua
013.634  000.021  000.021: sourcing /home/neimog/.local/share/nvim/lazy/LuaSnip/plugin/luasnip.vim
013.847  000.071  000.071: require('cmp.utils.api')
013.896  000.022  000.022: require('cmp.types.cmp')
013.960  000.031  000.031: require('cmp.utils.misc')
014.030  000.132  000.101: require('cmp.types.lsp')
014.067  000.035  000.035: require('cmp.types.vim')
014.069  000.220  000.031: require('cmp.types')
014.090  000.020  000.020: require('cmp.utils.highlight')
014.156  000.020  000.020: require('cmp.utils.debug')
014.161  000.049  000.029: require('cmp.utils.autocmd')
014.346  000.603  000.244: sourcing /home/neimog/.local/share/nvim/lazy/nvim-cmp/plugin/cmp.lua
014.572  000.033  000.033: sourcing /home/neimog/.local/share/nvim/lazy/nvim-colorizer.lua/plugin/colorizer.lua
015.688  000.036  000.036: require('vim.treesitter.language')
015.699  000.146  000.110: require('vim.treesitter.query')
016.081  000.062  000.062: require('vim.treesitter.languagetree')
016.108  000.170  000.108: require('vim.treesitter')
016.232  000.448  000.277: require('nvim-treesitter.parsers')
016.285  000.051  000.051: require('nvim-treesitter.utils')
016.290  000.551  000.052: require('nvim-treesitter.ts_utils')
016.294  000.594  000.043: require('nvim-treesitter.tsrange')
016.328  000.032  000.032: require('nvim-treesitter.caching')
016.340  000.842  000.069: require('nvim-treesitter.query')
016.352  000.938  000.096: require('nvim-treesitter.configs')
016.356  001.416  000.478: require('nvim-treesitter-textobjects')
016.578  000.082  000.082: require('nvim-treesitter.info')
016.659  000.078  000.078: require('nvim-treesitter.shell_command_selectors')
016.690  000.276  000.117: require('nvim-treesitter.install')
016.745  000.053  000.053: require('nvim-treesitter.statusline')
016.804  000.057  000.057: require('nvim-treesitter.query_predicates')
016.807  000.450  000.063: require('nvim-treesitter')
016.933  000.062  000.062: require('nvim-treesitter.textobjects.shared')
016.938  000.124  000.062: require('nvim-treesitter.textobjects.select')
017.142  000.098  000.098: require('nvim-treesitter.textobjects.attach')
017.200  000.056  000.056: require('nvim-treesitter.textobjects.repeatable_move')
017.208  000.225  000.072: require('nvim-treesitter.textobjects.move')
017.412  000.054  000.054: require('nvim-treesitter.textobjects.swap')
017.500  000.054  000.054: require('nvim-treesitter.textobjects.lsp_interop')
017.516  002.586  000.263: sourcing /home/neimog/.local/share/nvim/lazy/nvim-treesitter-textobjects/plugin/nvim-treesitter-textobjects.vim
018.340  000.768  000.768: sourcing /home/neimog/.local/share/nvim/lazy/nvim-treesitter/plugin/nvim-treesitter.lua
018.534  000.025  000.025: sourcing /home/neimog/.local/share/nvim/lazy/todo-comments.nvim/plugin/todo.vim
019.234  000.061  000.061: require('todo-comments.util')
019.259  000.187  000.126: require('todo-comments.config')
019.403  000.100  000.100: require('todo-comments.highlight')
019.407  000.146  000.046: require('todo-comments.jump')
019.410  000.856  000.523: require('todo-comments')
019.639  000.039  000.039: sourcing /home/neimog/.local/share/nvim/lazy/onedarkpro.nvim/plugin/onedarkpro.vim
022.558  000.136  000.136: require('vim.lsp.log')
022.810  000.006  000.006: require('vim.F')
022.814  000.247  000.241: require('vim.lsp.protocol')
023.054  000.127  000.127: require('vim.lsp._snippet')
023.120  000.063  000.063: require('vim.highlight')
023.134  000.319  000.129: require('vim.lsp.util')
023.149  001.255  000.553: require('vim.lsp.handlers')
023.246  000.096  000.096: require('vim.lsp.rpc')
023.292  000.043  000.043: require('vim.lsp.sync')
023.426  000.132  000.132: require('vim.lsp.buf')
023.473  000.044  000.044: require('vim.lsp.diagnostic')
023.536  000.060  000.060: require('vim.lsp.codelens')
023.639  002.214  000.584: require('vim.lsp')
023.713  002.553  000.339: sourcing /home/neimog/.local/share/nvim/lazy/nvim-lspconfig/plugin/lspconfig.lua
024.175  000.020  000.020: sourcing /home/neimog/.local/share/nvim/lazy/tabby.nvim/plugin/tabby.vim
024.816  000.184  000.184: sourcing /home/neimog/.local/share/nvim/lazy/telescope.nvim/plugin/telescope.lua
025.937  000.027  000.027: require('Comment.config')
026.092  000.063  000.063: require('Comment.ft')
026.096  000.157  000.094: require('Comment.utils')
026.136  000.038  000.038: require('Comment.opfunc')
026.172  000.035  000.035: require('Comment.extra')
026.176  000.610  000.353: require('Comment.api')
026.277  000.742  000.132: sourcing /home/neimog/.local/share/nvim/lazy/Comment.nvim/plugin/Comment.lua
026.349  000.034  000.034: require('Comment')
027.872  000.045  000.045: require('indent_blankline/utils')
027.876  000.731  000.686: require('indent_blankline')
028.123  000.053  000.053: require('indent_blankline.commands')
028.181  001.111  000.326: sourcing /home/neimog/.local/share/nvim/lazy/indent-blankline.nvim/plugin/indent_blankline.vim
028.556  000.038  000.038: sourcing /home/neimog/.local/share/nvim/lazy/trouble.nvim/plugin/trouble.vim
029.524  000.122  000.122: require('trouble.config')
029.528  000.161  000.039: require('trouble.util')
029.567  000.037  000.037: require('trouble.providers.qf')
029.612  000.043  000.043: require('trouble.providers.telescope')
029.656  000.042  000.042: require('trouble.providers.lsp')
029.694  000.036  000.036: require('trouble.providers.diagnostic')
029.697  000.552  000.233: require('trouble.providers')
029.729  000.030  000.030: require('trouble.text')
029.758  000.027  000.027: require('trouble.folds')
029.761  000.650  000.040: require('trouble.renderer')
029.775  000.757  000.106: require('trouble.view')
029.810  000.034  000.034: require('trouble.colors')
030.222  000.364  000.364: require('vim.diagnostic')
030.251  001.677  000.523: require('trouble')
030.842  000.073  000.073: sourcing /home/neimog/.local/share/nvim/lazy/ChatGPT.nvim/plugin/chatgpt.lua
031.582  000.079  000.079: require('nui.object')
031.874  000.061  000.061: require('nui.utils')
031.883  000.108  000.047: require('nui.text')
031.891  000.177  000.069: require('nui.line')
032.005  000.362  000.184: require('nui.popup.border')
032.064  000.044  000.044: require('nui.utils.buf_storage')
032.154  000.088  000.088: require('nui.utils.autocmd')
032.199  000.043  000.043: require('nui.utils.keymap')
032.250  000.049  000.049: require('nui.layout.utils')
032.272  000.688  000.102: require('nui.popup')
032.390  000.056  000.056: require('nui.split.utils')
032.429  000.155  000.099: require('nui.split')
032.479  000.048  000.048: require('nui.layout.float')
032.526  000.046  000.046: require('nui.layout.split')
032.539  001.175  000.159: require('nui.layout')
032.738  000.157  000.157: require('nui.input')
032.760  000.220  000.063: require('chatgpt.input')
033.039  000.173  000.173: require('chatgpt.config')
033.124  000.083  000.083: require('chatgpt.utils')
033.267  000.141  000.141: require('chatgpt.spinner')
033.485  000.073  000.073: require('chatgpt.common.classes')
033.537  000.050  000.050: require('chatgpt.common.input_widget')
033.669  000.027  000.027: require('plenary.bit')
033.696  000.025  000.025: require('plenary.functional')
033.711  000.172  000.120: require('plenary.path')
033.778  000.066  000.066: require('plenary.scandir')
033.784  000.514  000.153: require('chatgpt.flows.chat.session')
033.810  000.026  000.026: require('chatgpt.flows.chat.tokens')
033.815  001.053  000.116: require('chatgpt.chat')
033.887  000.044  000.044: require('plenary.job')
033.891  000.075  000.031: require('chatgpt.api')
034.527  000.049  000.049: require('telescope._extensions')
034.531  000.101  000.052: require('telescope')
034.656  000.023  000.023: require('plenary.tbl')
034.659  000.049  000.026: require('plenary.vararg.rotate')
034.660  000.074  000.026: require('plenary.vararg')
034.684  000.023  000.023: require('plenary.errors')
034.688  000.127  000.030: require('plenary.async.async')
034.792  000.024  000.024: require('plenary.async.structs')
034.798  000.058  000.033: require('plenary.async.control')
034.802  000.087  000.030: require('plenary.async.util')
034.804  000.115  000.027: require('plenary.async.tests')
034.805  000.273  000.031: require('plenary.async')
034.920  000.030  000.030: require('plenary.strings')
034.923  000.080  000.050: require('plenary.window.border')
034.950  000.026  000.026: require('plenary.window')
034.986  000.035  000.035: require('plenary.popup.utils')
034.990  000.183  000.042: require('plenary.popup')
035.283  000.074  000.074: require('telescope.deprecated')
035.593  000.194  000.194: require('plenary.log')
035.637  000.282  000.089: require('telescope.log')
035.862  000.054  000.054: require('telescope.state')
035.869  000.184  000.130: require('telescope.utils')
035.877  000.593  000.127: require('telescope.sorters')
037.576  002.463  001.796: require('telescope.config')
037.682  000.097  000.097: require('telescope.pickers.scroller')
037.726  000.042  000.042: require('telescope.actions.state')
037.772  000.044  000.044: require('telescope.actions.utils')
037.884  000.051  000.051: require('telescope.actions.mt')
037.897  000.123  000.072: require('telescope.actions.set')
038.045  000.102  000.102: require('telescope.config.resolve')
038.047  000.149  000.047: require('telescope.pickers.entry_display')
038.107  000.058  000.058: require('telescope.from_entry')
038.355  003.364  000.388: require('telescope.actions')
038.420  000.063  000.063: require('telescope.debounce')
038.510  000.088  000.088: require('telescope.mappings')
038.562  000.050  000.050: require('telescope.pickers.highlights')
038.601  000.037  000.037: require('telescope.pickers.window')
038.701  000.045  000.045: require('telescope.algos.linked_list')
038.704  000.101  000.056: require('telescope.entry_manager')
038.742  000.036  000.036: require('telescope.pickers.multi')
038.758  004.837  000.542: require('telescope.pickers')
038.847  000.040  000.040: require('telescope.previewers.previewer')
038.944  000.096  000.096: require('telescope.previewers.term_previewer')
039.188  000.028  000.028: require('plenary.context_manager')
039.192  000.080  000.052: require('telescope.previewers.utils')
040.130  000.936  000.936: require('plenary.filetype')
040.178  001.232  000.215: require('telescope.previewers.buffer_previewer')
040.182  001.423  000.056: require('telescope.previewers')
040.199  006.308  000.047: require('chatgpt.prompts')
040.328  000.073  000.073: require('chatgpt.settings')
040.349  000.148  000.075: require('chatgpt.code_edits')
040.411  000.060  000.060: require('chatgpt.flows.chat.sessions')
040.607  000.041  000.041: require('chatgpt.signs')
040.615  000.100  000.059: require('chatgpt.flows.actions.base')
040.619  000.153  000.053: require('chatgpt.flows.actions.completions')
040.674  000.054  000.054: require('chatgpt.flows.actions.edits')
040.678  000.265  000.058: require('chatgpt.flows.actions')
040.741  000.062  000.062: require('chatgpt.flows.code_completions')
040.747  009.446  000.081: require('chatgpt.module')
040.749  009.887  000.440: require('chatgpt')
041.893  000.952  000.952: sourcing /home/neimog/.local/share/nvim/lazy/vim-fugitive/plugin/fugitive.vim
041.953  000.010  000.010: sourcing /home/neimog/.local/share/nvim/lazy/vim-fugitive/ftdetect/fugitive.vim
042.250  000.129  000.129: sourcing /usr/share/nvim/runtime/plugin/gzip.vim
042.280  000.009  000.009: sourcing /usr/share/nvim/runtime/plugin/health.vim
042.369  000.061  000.061: sourcing /usr/share/nvim/runtime/plugin/man.lua
042.792  000.140  000.140: sourcing /usr/share/nvim/runtime/pack/dist/opt/matchit/plugin/matchit.vim
042.847  000.452  000.312: sourcing /usr/share/nvim/runtime/plugin/matchit.vim
043.049  000.161  000.161: sourcing /usr/share/nvim/runtime/plugin/matchparen.vim
043.360  000.275  000.275: sourcing /usr/share/nvim/runtime/plugin/netrwPlugin.vim
043.497  000.006  000.006: sourcing /home/neimog/.local/share/nvim/rplugin.vim
043.503  000.113  000.107: sourcing /usr/share/nvim/runtime/plugin/rplugin.vim
043.572  000.046  000.046: sourcing /usr/share/nvim/runtime/plugin/shada.vim
043.627  000.019  000.019: sourcing /usr/share/nvim/runtime/plugin/spellfile.vim
043.735  000.084  000.084: sourcing /usr/share/nvim/runtime/plugin/tarPlugin.vim
043.823  000.060  000.060: sourcing /usr/share/nvim/runtime/plugin/tohtml.vim
043.860  000.014  000.014: sourcing /usr/share/nvim/runtime/plugin/tutor.vim
044.057  000.174  000.174: sourcing /usr/share/nvim/runtime/plugin/zipPlugin.vim
044.674  000.069  000.069: require('cmp.utils.char')
044.682  000.154  000.085: require('cmp.utils.str')
044.719  000.036  000.036: require('cmp.utils.pattern')
044.868  000.037  000.037: require('cmp.utils.buffer')
044.894  000.127  000.090: require('cmp.utils.keymap')
044.898  000.177  000.049: require('cmp.utils.feedkeys')
044.954  000.055  000.055: require('cmp.utils.async')
045.077  000.034  000.034: require('cmp.utils.cache')
045.081  000.100  000.066: require('cmp.context')
045.275  000.068  000.068: require('cmp.config.mapping')
045.404  000.073  000.073: require('cmp.config.compare')
045.407  000.129  000.056: require('cmp.config.default')
045.422  000.281  000.084: require('cmp.config')
045.588  000.087  000.087: require('cmp.matcher')
045.598  000.175  000.087: require('cmp.entry')
045.606  000.524  000.068: require('cmp.source')
045.704  000.042  000.042: require('cmp.utils.event')
045.853  000.043  000.043: require('cmp.utils.options')
045.859  000.105  000.063: require('cmp.utils.window')
045.861  000.155  000.050: require('cmp.view.docs_view')
045.980  000.118  000.118: require('cmp.view.custom_entries_view')
046.084  000.101  000.101: require('cmp.view.wildmenu_entries_view')
046.145  000.059  000.059: require('cmp.view.native_entries_view')
046.194  000.047  000.047: require('cmp.view.ghost_text_view')
046.206  000.599  000.075: require('cmp.view')
046.351  001.926  000.282: require('cmp.core')
046.482  000.040  000.040: require('cmp.config.sources')
046.519  000.033  000.033: require('cmp.config.window')
046.563  002.218  000.219: require('cmp')
046.625  000.060  000.060: require('cmp_luasnip')
046.661  002.358  000.080: sourcing /home/neimog/.local/share/nvim/lazy/cmp_luasnip/after/plugin/cmp_luasnip.lua
046.909  000.052  000.052: require('cmp_nvim_lsp.source')
046.913  000.111  000.059: require('cmp_nvim_lsp')
046.926  000.201  000.090: sourcing /home/neimog/.local/share/nvim/lazy/cmp-nvim-lsp/after/plugin/cmp_nvim_lsp.lua
051.933  000.225  000.225: require('onedarkpro.utils')
052.337  000.391  000.391: require('onedarkpro.config')
052.341  000.874  000.258: require('onedarkpro')
053.783  001.295  001.295: require('onedarkpro.lib.hash')
059.387  003.212  003.212: sourcing /home/neimog/.local/share/nvim/lazy/onedarkpro.nvim/colors/onelight.vim
059.690  000.069  000.069: require('indent_blankline.utils')
067.336  000.053  000.053: require('harpoon.utils')
067.484  000.142  000.142: require('harpoon.dev')
067.777  000.635  000.440: require('harpoon')
067.791  000.919  000.283: require('harpoon.mark')
067.864  000.071  000.071: require('harpoon.ui')
068.242  000.036  000.036: require('notify.util.queue')
068.246  000.096  000.060: require('notify.util')
068.353  000.105  000.105: require('notify.config.highlights')
068.358  000.296  000.095: require('notify.config')
068.394  000.034  000.034: require('notify.stages')
068.497  000.101  000.101: require('notify.service.notification')
068.648  000.047  000.047: require('notify.animate.spring')
068.650  000.079  000.031: require('notify.animate')
068.654  000.154  000.076: require('notify.windows')
068.775  000.039  000.039: require('notify.service.buffer.highlights')
068.780  000.088  000.049: require('notify.service.buffer')
068.782  000.127  000.039: require('notify.service')
068.827  000.043  000.043: require('notify.stages.util')
068.833  000.903  000.147: require('notify')
068.879  000.029  000.029: require('notify.stages.static')
069.242  000.174  000.174: require('telescope.make_entry')
069.293  000.047  000.047: require('telescope.finders.async_static_finder')
069.438  000.039  000.039: require('plenary.class')
069.470  000.132  000.093: require('telescope._')
069.472  000.177  000.045: require('telescope.finders.async_oneshot_finder')
069.513  000.040  000.040: require('telescope.finders.async_job_finder')
069.518  000.577  000.138: require('telescope.finders')
069.529  000.630  000.054: require('telescope._extensions.notify')
069.665  000.028  000.028: require('toggleterm.lazy')
069.691  000.024  000.024: require('toggleterm.constants')
069.804  000.111  000.111: require('toggleterm.terminal')
069.810  000.249  000.086: require('toggleterm')
069.885  000.032  000.032: require('toggleterm.colors')
069.922  000.035  000.035: require('toggleterm.utils')
070.015  000.203  000.135: require('toggleterm.config')
070.212  000.050  000.050: require('toggleterm.commandline')
070.385  000.044  000.044: require('telescope.themes')
070.588  000.074  000.074: require('telescope.builtin.__lsp')
070.598  000.160  000.086: require('telescope.builtin')
070.703  000.057  000.057: require('telescope._extensions.project.utils')
070.757  000.052  000.052: require('plenary.iterators')
070.760  000.160  000.052: require('telescope._extensions.project.git')
070.791  000.404  000.084: require('telescope._extensions.project.actions')
070.838  000.045  000.045: require('telescope._extensions.project.finders')
070.840  000.555  000.061: require('telescope._extensions.project.main')
070.928  000.692  000.137: require('telescope._extensions.project')
071.076  000.143  000.143: require('alpha')
071.166  000.086  000.086: require('alpha.themes.dashboard')
071.642  000.035  000.035: require('gitsigns.async')
071.745  000.026  000.026: require('gitsigns.message')
071.758  000.112  000.086: require('gitsigns.config')
071.791  000.032  000.032: require('gitsigns.debug.log')
071.825  000.032  000.032: require('gitsigns.uv')
071.833  000.278  000.067: require('gitsigns')
072.951  000.115  000.115: require('fzf_lib')
072.963  000.207  000.092: require('telescope._extensions.fzf')
074.141  000.098  000.098: require('telescope._extensions.file_browser.utils')
074.177  000.292  000.195: require('telescope._extensions.file_browser.actions')
074.400  000.055  000.055: require('telescope._extensions.file_browser.git')
074.412  000.155  000.100: require('telescope._extensions.file_browser.make_entry')
074.430  000.252  000.097: require('telescope._extensions.file_browser.finders')
074.471  000.040  000.040: require('telescope._extensions.file_browser.picker')
074.519  000.046  000.046: require('telescope._extensions.file_browser.config')
074.522  000.683  000.054: require('telescope._extensions.file_browser')
076.309  000.047  000.047: require('nvim-treesitter.highlight')
076.499  000.055  000.055: require('nvim-treesitter.locals')
076.504  000.107  000.052: require('nvim-treesitter.incremental_selection')
076.586  000.048  000.048: require('nvim-treesitter.indent')
076.621  000.032  000.032: require('neodev')
076.656  000.033  000.033: require('neodev.config')
076.730  000.031  000.031: require('neodev.util')
076.732  000.070  000.039: require('neodev.lsp')
076.829  000.095  000.095: require('lspconfig.util')
077.260  000.029  000.029: require('mason-core.path')
077.271  000.074  000.045: require('mason.settings')
077.379  000.063  000.063: require('mason-core.functional')
077.443  000.024  000.024: require('mason-core.functional.data')
077.447  000.060  000.036: require('mason-core.functional.function')
077.490  000.029  000.029: require('mason-core.functional.relation')
077.525  000.029  000.029: require('mason-core.functional.logic')
077.535  000.263  000.082: require('mason-core.platform')
077.537  000.380  000.042: require('mason')
077.645  000.043  000.043: require('mason-core.functional.list')
077.679  000.031  000.031: require('mason-core.functional.string')
077.698  000.146  000.071: require('mason.api.command')
077.738  000.036  000.036: require('mason-registry.sources')
077.826  000.048  000.048: require('mason-core.log')
077.830  000.087  000.039: require('mason-lspconfig')
077.887  000.028  000.028: require('mason-lspconfig.settings')
077.926  000.033  000.033: require('mason-lspconfig.lspconfig_hook')
078.098  000.073  000.073: require('mason-core.functional.table')
078.139  000.211  000.138: require('mason-lspconfig.mappings.server')
078.248  000.042  000.042: require('mason-core.async')
078.274  000.024  000.024: require('mason-core.async.uv')
078.279  000.107  000.041: require('mason-core.fs')
078.311  000.030  000.030: require('mason-core.optional')
078.339  000.027  000.027: require('mason-core.EventEmitter')
078.352  000.210  000.046: require('mason-registry')
078.379  000.025  000.025: require('mason-lspconfig.server_config_extensions')
078.423  000.043  000.043: require('lspconfig.configs')
078.484  000.039  000.039: require('lspconfig.server_configurations.omnisharp')
078.577  000.023  000.023: require('mason-core.notify')
078.579  000.055  000.032: require('mason-lspconfig.ensure_installed')
079.797  000.074  000.074: require('mason-core.result')
080.103  000.192  000.192: require('mason-core.process')
080.136  000.276  000.084: require('mason-core.spawn')
080.200  000.062  000.062: require('mason-core.receipt')
080.220  000.420  000.082: require('mason-core.installer.context')
080.298  000.076  000.076: require('mason-core.installer.linker')
080.336  000.037  000.037: require('mason-core.async.control')
080.344  000.750  000.143: require('mason-core.installer')
080.407  000.062  000.062: require('mason-core.installer.handle')
080.653  000.067  000.067: require('mason-core.managers.powershell')
080.655  000.116  000.049: require('mason-core.fetch')
080.658  000.153  000.037: require('mason-core.managers.cargo.client')
080.788  000.073  000.073: require('mason-core.managers.std')
080.878  000.039  000.039: require('mason-core.providers')
080.891  000.101  000.062: require('mason-core.managers.github.client')
080.897  000.238  000.064: require('mason-core.managers.github')
080.911  000.459  000.068: require('mason-core.managers.cargo')
081.014  000.102  000.102: require('mason-core.managers.composer')
081.139  000.122  000.122: require('mason-core.managers.gem')
081.220  000.078  000.078: require('mason-core.managers.git')
081.324  000.102  000.102: require('mason-core.managers.go')
081.402  000.077  000.077: require('mason-core.managers.luarocks')
081.478  000.073  000.073: require('mason-core.managers.npm')
081.566  000.087  000.087: require('mason-core.managers.pip3')
081.576  001.166  000.065: require('mason-core.package.version-check')
081.583  003.003  001.025: require('mason-core.package')
081.638  000.044  000.044: require('mason-registry.sources.lua')
081.779  000.138  000.138: require('mason-registry.index')
081.900  000.118  000.118: require('mason-registry.index.pyright')
082.137  000.109  000.109: require('mason-registry.index.clangd')
082.328  000.081  000.081: require('mason-core.functional.number')
082.378  000.235  000.154: require('mason-lspconfig.api.command')
082.544  000.081  000.081: require('lspconfig')
082.648  000.097  000.097: require('lspconfig.server_configurations.pyright')
083.502  000.137  000.137: require('lspconfig.server_configurations.clangd')
084.461  000.303  000.303: require('lspconfig/util')
084.591  000.127  000.127: require('py_lsp.options')
084.649  000.055  000.055: require('py_lsp.utils')
084.700  000.049  000.049: require('py_lsp.commands')
084.818  000.055  000.055: require('py_lsp.python.strategies')
084.843  000.140  000.086: require('py_lsp.python')
084.903  000.058  000.058: require('py_lsp.lsp')
084.910  000.911  000.179: require('py_lsp')
086.044  000.199  000.199: require('fidget.log')
086.063  000.338  000.139: require('fidget')
086.217  000.110  000.110: require('fidget.spinners')
087.648  000.093  000.093: require('lualine_require')
087.923  000.480  000.388: require('lualine')
089.125  000.108  000.108: require('onedarkpro.lib.color')
089.130  000.197  000.089: require('onedarkpro.helpers')
089.173  000.041  000.041: require('onedarkpro.theme')
089.229  000.053  000.053: require('onedarkpro.themes.onelight')
089.273  000.043  000.043: require('onedarkpro.lib.palette')
094.747  000.101  000.101: require('lualine.utils.mode')
097.418  000.044  000.044: require('notify.render')
097.476  000.024  000.024: require('notify.render.base')
097.479  000.057  000.033: require('notify.render.simple')
099.446  000.039  000.039: require('luasnip.loaders.util')
099.454  000.099  000.060: require('luasnip.loaders.from_lua')
099.771  000.034  000.034: require('luasnip.nodes.util')
099.798  000.024  000.024: require('luasnip.util.events')
099.806  000.158  000.100: require('luasnip.nodes.node')
099.934  000.126  000.126: require('luasnip.nodes.insertNode')
100.014  000.077  000.077: require('luasnip.nodes.textNode')
100.076  000.058  000.058: require('luasnip.util.mark')
100.148  000.031  000.031: require('luasnip.util.pattern_tokenizer')
100.173  000.024  000.024: require('luasnip.util.dict')
100.215  000.637  000.163: require('luasnip.nodes.snippet')
100.329  000.041  000.041: require('luasnip.util.parser.neovim_ast')
100.363  000.032  000.032: require('luasnip.util.str')
100.513  000.031  000.031: require('luasnip.util.directed_graph')
100.519  000.265  000.161: require('luasnip.util.parser.ast_utils')
100.583  000.062  000.062: require('luasnip.nodes.functionNode')
100.661  000.076  000.076: require('luasnip.nodes.choiceNode')
100.730  000.067  000.067: require('luasnip.nodes.dynamicNode')
100.762  000.030  000.030: require('luasnip.util.functions')
100.767  000.550  000.050: require('luasnip.util.parser.ast_parser')
100.835  000.067  000.067: require('luasnip.util.parser.neovim_parser')
100.840  001.295  000.041: require('luasnip.util.parser')
100.842  001.326  000.032: require('luasnip.nodes.snippetProxy')
100.847  001.372  000.046: require('luasnip.loaders.from_snipmate')
101.060  000.153  000.153: require('luasnip.util.jsonc')
101.066  000.199  000.046: require('luasnip.loaders.from_vscode')
103.263  000.368  000.368: require('telescope.builtin.__files')
103.653  100.425  046.774: sourcing /home/neimog/.config/nvim/init.lua
103.672  000.618: sourcing vimrc file(s)
104.070  000.139  000.139: sourcing /usr/share/nvim/runtime/filetype.lua
104.152  000.033  000.033: sourcing /usr/share/nvim/runtime/filetype.vim
104.424  000.085  000.085: sourcing /usr/share/nvim/runtime/syntax/synload.vim
105.261  001.061  000.976: sourcing /usr/share/nvim/runtime/syntax/syntax.vim
105.277  000.372: inits 3
107.657  002.381: reading ShaDa
108.301  000.644: opening buffers
108.483  000.181: BufEnter autocommands
108.486  000.003: editing files in windows
109.687  001.201: VimEnter autocommands
109.722  000.035: UIEnter autocommands
109.724  000.002: before starting main loop
111.190  001.466: first screen update
111.192  000.003: --- NVIM STARTED ---
