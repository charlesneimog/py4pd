#include "py4pd.h"

// ====================================================
/*
* @brief This function parse the arguments for pd Objects created with the library
* @param x is the py4pd object
* @param code is the code object of the function
* @param argc is the number of arguments
* @param argv is the arguments
* @return 1 if all arguments are ok, 0 if not
*/
int Py4pdUtils_ParseLibraryArguments(t_py *x, PyCodeObject *code, int argc, t_atom *argv){
    int argsNumberDefined = 0;
    x->x_numOutlets = -1;
    x->n_channels = 1;
    int i, j;
    for (i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            if (strcmp(argv[i].a_w.w_symbol->s_name, "-n_args") == 0 || strcmp(argv[i].a_w.w_symbol->s_name, "-a") == 0) {
                if (i + 1 < argc) {
                    if (argv[i + 1].a_type == A_FLOAT) {
                        x->py_arg_numbers = (int)argv[i + 1].a_w.w_float;
                        argsNumberDefined = 1;
                        // remove -n_args and the number of arguments from the arguments list
                        for (j = i; j < argc; j++) {
                            argv[j] = argv[j + 2];
                        }
                    }
                }
            }
            else if (strcmp(argv[i].a_w.w_symbol->s_name, "-outn") == 0){
                if (argv[i + 1].a_type == A_FLOAT) {
                    x->x_numOutlets = (int)argv[i + 1].a_w.w_float - 1;
                    // remove -outn and the number of outlets from the arguments list
                    for (j = i; j < argc; j++) {
                        argv[j] = argv[j + 2];
                    }
                }
                else{
                    x->x_numOutlets = -1; // -1 means that the number of outlets is not defined
                }
            }
            else if (strcmp(argv[i].a_w.w_symbol->s_name, "-ch") == 0 || strcmp(argv[i].a_w.w_symbol->s_name, "-channels") == 0){
                if (argv[i + 1].a_type == A_FLOAT) {
                    x->n_channels = (int)argv[i + 1].a_w.w_float;
                    for (j = i; j < argc; j++) {
                        argv[j] = argv[j + 2];
                    }
                }
            }
        }
    }
    if (code->co_flags & CO_VARARGS) {
        if (argsNumberDefined == 0) {
            pd_error(x, "[%s] this function uses *args, you need to specify the number of arguments using -n_args (-a for short) {number}", x->objectName->s_name);
            return 0;
        }
    }
    if (code->co_flags & CO_VARKEYWORDS) {
        x->kwargs = 1;
    }
    if (code->co_argcount != 0){
        if (x->py_arg_numbers == 0) {
            x->py_arg_numbers = code->co_argcount;
        }
        else{
            x->py_arg_numbers = x->py_arg_numbers + code->co_argcount;
        }
    }
    return 1; 
}


// ====================================================
/*
* @brief This function parse the arguments for the py4pd object
* @param x is the py4pd object
* @param c is the canvas of the object
* @param argc is the number of arguments
* @param argv is the arguments
* @return return void
*/

void Py4pdUtils_ParseArguments(t_py *x, t_canvas *c, int argc, t_atom *argv) {

    int i;
    for (i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            t_symbol *py4pdArgs = atom_getsymbolarg(i, argc, argv);
            if (py4pdArgs == gensym("-picture") ||
                py4pdArgs == gensym("-score") ||
                py4pdArgs == gensym("-pic") ||
                py4pdArgs == gensym("-canvas")) {
                Py4pdPic_InitVisMode(x, c, py4pdArgs, i, argc, argv, NULL);
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
            } 
            else if (py4pdArgs == gensym("-nvim") ||
                        py4pdArgs == gensym("-vscode") ||
                        py4pdArgs == gensym("-sublime") || 
                        py4pdArgs == gensym("-emacs")) {
                // remove the '-' from the name of the editor
                const char *editor = py4pdArgs->s_name;
                editor++;
                x->editorName = gensym(editor); 
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
            } 
            else if (py4pdArgs == gensym("-audioin")) {
                x->audioInput = 1;
                x->audioOutput = 0;
                x->use_NumpyArray = 0;
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
            } 
            else if (py4pdArgs == gensym("-audioout")) {
                // post("[py4pd] Audio Outlets enabled");
                x->audioOutput = 1;
                x->audioInput = 0;
                x->use_NumpyArray = 0;
                x->out1 = outlet_new(&x->obj, gensym("signal"));  // create a signal outlet
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
            }
            else if (py4pdArgs == gensym("-audio")) {
                x->audioInput = 1;
                x->audioOutput = 1;
                x->out1 = outlet_new(
                    &x->obj, gensym("signal"));  // create a signal outlet
                x->use_NumpyArray = 0;
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
            }
        }
    }
    if (x->audioOutput == 0) {
        x->out1 = outlet_new(&x->obj, 0);  // cria um outlet caso o objeto nao contenha audio
    }
}

// ====================================================
// ============================================
/**
 * @brief Free the memory of the object
 * @param x 
 * @return void* 
 */
void *Py4pdLib_FreeObj(t_py *x) {
    object_count--;
    if (object_count == 0) {
        object_count = 0;
        char command[1000];
        #ifdef _WIN64
            sprintf(command, "del /q /s %s\\*", x->tempPath->s_name);
            Py4pdUtils_ExecuteSystemCommand(command);
        #else
            sprintf(command, "rm -rf %s", x->tempPath->s_name);
            Py4pdUtils_ExecuteSystemCommand(command);
        #endif
    }
    if (x->visMode != 0) {
        Py4pdPic_Free(x);
    }
    if (x->pdcollect != NULL) {
        // free_table(x->pdcollect);
    }



    return NULL;
}

// ====================================================
/*
* @brief get the folder name of something
* @param x is the py4pd object
* @return save the py4pd folder in x->py4pdPath
*/
char* Py4pdUtils_GetFolderName(char* path) {
    char* folder = NULL;
    char* last_separator = NULL;

    // Find the last occurrence of a path separator
    #ifdef _WIN32
        last_separator = strrchr(path, '\\');
    #else
        last_separator = strrchr(path, '/');
    #endif

    // If a separator is found, extract the folder name
    if (last_separator != NULL) {
        size_t folder_length = last_separator - path;
        folder = malloc(folder_length + 1);
        strncpy(folder, path, folder_length);
        folder[folder_length] = '\0';
    }

    return folder;
}

// ====================================================
/*
* @brief get the folder name of something
* @param x is the py4pd object
* @return save the py4pd folder in x->py4pdPath
*/

const char* Py4pdUtils_GetFilename(const char* path) {
    const char* filename = NULL;

    // Find the last occurrence of a path separator
    const char* last_separator = strrchr(path, '/');

    #ifdef _WIN32
        const char* last_separator_win = strrchr(path, '\\');
        if (last_separator_win != NULL && last_separator_win > last_separator) {
            last_separator = last_separator_win;
        }
    #endif

    // If a separator is found, extract the filename
    if (last_separator != NULL) {
        filename = last_separator + 1;
    } else {
        // No separator found, use the entire path as the filename
        filename = path;
    }

    // remove .py from filename
    const char* last_dot = strrchr(filename, '.');
    if (last_dot != NULL) {
        size_t filename_length = last_dot - filename;
        char* filename_without_extension = malloc(filename_length + 1);
        strncpy(filename_without_extension, filename, filename_length);
        filename_without_extension[filename_length] = '\0';
        filename = filename_without_extension;
    }

    return filename;
}

// ====================================================
void Py4pdUtils_CheckPkgNameConflict(t_py *x, char *folderToCheck, t_symbol *script_file_name){
        DIR *dir;
        struct dirent *entry;

        if ((dir = opendir(folderToCheck)) == NULL) {
            return;
        }
        dir = opendir(folderToCheck);
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
                if (strcmp(entry->d_name, script_file_name->s_name) == 0) {
                    post("");
                    pd_error(x, "[py4pd] The library '%s' conflicts with a Python package name.", script_file_name->s_name);
                    pd_error(x, "[py4pd] This can cause problems related with py4pdLoadObjects.");
                    pd_error(x, "[py4pd] Rename the library.");
                    post("");
                }
            }
        }
        closedir(dir);
    return;
}

// ====================================================
/*
* @brief Get the py4pd folder object, it creates the folder for scripts inside resources
* @param x is the py4pd object
* @return save the py4pd folder in x->py4pdPath
*/

void Py4pdUtils_FindObjFolder(t_py *x){
    int libraryNotFound = 1;

    for (int i = 0; 1; i++){ 
        const char *pathelem = namelist_get(STUFF->st_searchpath, i);
        if (!pathelem){
            break;
        }
        char library_path[MAXPDSTRING];
        #ifdef _WIN64
            snprintf(library_path, MAXPDSTRING, "%s\\%s\\", pathelem, "py4pd");
        #else
            snprintf(library_path, MAXPDSTRING, "%s/%s/", pathelem, "py4pd"); 
        #endif
        if (access(library_path, F_OK) != -1) {
            libraryNotFound = 0;
            x->py4pdPath = gensym(library_path);
            return;
        }
    }
    if (libraryNotFound){
        pd_error(x, "[py4pd] py4pd was not found in Search Path, this causes instabilities.");
        x->py4pdPath = canvas_getdir(x->canvas);
        return;
    }
}

// ===================================================================
/**
 * @brief Get the temp path object (inside Users/.py4pd), it creates the folder if it not exist
 * @param x is the py4pd object
 * @return save the temp path in x->tempPath
 */

void Py4pdUtils_CreateTempFolder(t_py *x) {
    #ifdef _WIN64
        // get user folder
        char *user_folder = getenv("USERPROFILE");
        LPSTR home = (LPSTR)malloc(256 * sizeof(char));
        memset(home, 0, 256);
        sprintf(home, "%s\\.py4pd\\", user_folder);
        x->tempPath = gensym(home);
        if (access(home, F_OK) == -1) {
            char *command = (char *)malloc(256 * sizeof(char));
            memset(command, 0, 256);
            if (!CreateDirectory(home, NULL)){
                post("Failed to create directory, Report, this create instabilities: %d\n", GetLastError());
            }
            if (!SetFileAttributes(home, FILE_ATTRIBUTE_HIDDEN)){
                post("Failed to set hidden attribute: %d\n", GetLastError());
            }
            free(command);
        }
        free(home);
    #else
        const char *home = getenv("HOME");
        char *temp_folder = (char *)malloc(256 * sizeof(char));
        memset(temp_folder, 0, 256);
        sprintf(temp_folder, "%s/.py4pd/", home);
        x->tempPath = gensym(temp_folder);
        if (access(temp_folder, F_OK) == -1) {
            char *command = (char *)malloc(256 * sizeof(char));
            memset(command, 0, 256);
            sprintf(command, "mkdir -p %s", temp_folder);
            system(command);
            free(command);
        }
        free(temp_folder);

    #endif
}

// ===================================================================
/*
 * @brief It creates the commandline to open the editor
 * @param x is the py4pd object
 * @return the commandline to open the editor
 */
void Py4pdUtils_GetEditorCommand(t_py *x, char *command, int line) {
    const char *editor = x->editorName->s_name;
    const char *filename = x->script_name->s_name;
    const char *home = x->pdPatchFolder->s_name;
    char completePath[MAXPDSTRING];

    if (x->pyObject){
        sprintf(completePath, "'%s'", filename);
    }
    else{
        sprintf(completePath, "'%s/%s.py'", home, filename);
    }
    // check if there is .py in filename
    if (strcmp(editor, PY4PD_EDITOR) == 0) {
        sprintf(command, "%s %s", PY4PD_EDITOR, completePath);
    } 
    else if (strcmp(editor, "vscode") == 0) {
        sprintf(command, "code -g '%s:%d'", completePath, line);
    } 
    else if (strcmp(editor, "nvim") == 0) {
        // if it is linux
        #ifdef __linux__
            char *env_var = getenv("XDG_CURRENT_DESKTOP");
            if (env_var == NULL) {
                pd_error(x, "[py4pd] Your desktop environment is not supported, please report.");
                sprintf(command, "ls ");
            }
            else{
                if (strcmp(env_var, "GNOME") == 0) {
                    sprintf(command, "gnome-terminal -- nvim +%d %s", line, completePath);
                }
                else if (strcmp(env_var, "KDE") == 0) {
                    pd_error(x, "[py4pd] This is untested, please report if it works.");
                    sprintf(command, "konsole -e \"nvim +%d %s\"", line, completePath);
                }
                else{
                    pd_error(x, "[py4pd] Your desktop environment %s is not supported, please report.", env_var);
                    sprintf(command, "ls ");
                }
            }
        #else
            sprintf(command, "nvim +%d %s", line, completePath);
        #endif
    } 
    else if (strcmp(editor, "gvim") == 0) {
        sprintf(command, "gvim +%d %s", line, completePath);
    } 
    else if (strcmp(editor, "sublime") == 0) {
        sprintf(command, "subl --goto %s:%d", completePath, line);
    } 
    else if (strcmp(editor, "emacs") == 0) {
        sprintf(command, "emacs --eval '(progn (find-file \"%s\") (goto-line %d))'", completePath, line);

    } 
    else {
        pd_error(x, "[py4pd] editor %s not supported.", editor);
    }
    return; 
}

// ====================================
/*
* @brief Run system command and check for errors
* @param command is the command to run
* @return void, but it prints the error if it fails
*/

void Py4pdUtils_ExecuteSystemCommand(const char *command) {
    #ifdef _WIN64
        SHELLEXECUTEINFO sei = {0};
        sei.cbSize = sizeof(sei);
        sei.fMask = SEE_MASK_NOCLOSEPROCESS;
        sei.lpFile = "cmd.exe ";
        sei.lpParameters = command;
        sei.nShow = SW_HIDE;
        ShellExecuteEx(&sei);
        CloseHandle(sei.hProcess);
        if (sei.hProcess == NULL) {
            pd_error(NULL, "[py4pd] Failed to execute command: %s", command);
            return;
        }
        return;
    #else
        int result = system(command);
        if (result != 0) {
            pd_error(NULL, "[py4pd] Failed to execute command: %s", command);
            return;
        }
    #endif
}

// ============================================
/*
* @brief See if str is a number or a dot
* @param str is the string to check
* @return 1 if it is a number or a dot, 0 otherwise
*/
int Py4pdUtils_IsNumericOrDot(const char *str) {
    int hasDot = 0;
    while (*str) {
        if (isdigit(*str)) {
            str++;
        } else if (*str == '.' && !hasDot) {
            hasDot = 0;
            str++;
        } else {
            return 0;
        }
    }
    return 1;
}

// =====================================================================
/*
* @brief Remove some char from a string
* @param str is the string to remove the char
* @param c is the char to remove
* @return the string without the char
*/
void Py4pdUtils_RemoveChar(char *str, char c) {
    int i, j;
    for (i = 0, j = 0; str[i] != '\0'; i++) {
        if (str[i] != c) {
            str[j] = str[i];
            j++;
        }
    }
    str[j] = '\0';
}

// =====================================================================
/*
 * @brief Py4pdUtils_Mtok is a function separated from the tokens of one string
 * @param input is the string to be separated
 * @param delimiter is the string to be separated
 * @return the string separated by the delimiter
*/
char *Py4pdUtils_Mtok(char *input, char *delimiter) { 
    static char *string;
    if(input != NULL)
        string = input;
    if(string == NULL)
        return string;
    char *end = strstr(string, delimiter);
    while(end == string){
        *end = '\0';
        string = end + strlen(delimiter);
        end = strstr(string, delimiter);
    };
    if(end == NULL){
        char *temp = string;
        string = NULL;
        return temp;
    }
    char *temp = string;
    *end = '\0';
    string = end + strlen(delimiter);
    return(temp);
}

// =====================================================================
/*
* @brief Convert and output Python Values to PureData values
* @param x is the py4pd object
* @param pValue is the Python value to convert
* @return nothing, but output the value to the outlet
*/
void Py4pdUtils_FromSymbolSymbol(t_py *x, t_symbol *s, t_outlet *outlet){ 
    (void)x;
    //new and redone - Derek Kwan
    long unsigned int seplen = strlen(" ");
    seplen++;
    char *sep = t_getbytes(seplen * sizeof(*sep));
    memset(sep, '\0', seplen);
    strcpy(sep, " "); 
    if(s){
        long unsigned int iptlen = strlen(s->s_name);
        t_atom* out = t_getbytes(iptlen * sizeof(*out));
        iptlen++;
        char *newstr = t_getbytes(iptlen * sizeof(*newstr));
        memset(newstr, '\0', iptlen);
        strcpy(newstr, s->s_name);
        int atompos = 0; //position in atom
        char *ret = Py4pdUtils_Mtok(newstr, sep);
        char *err; // error pointer
        while(ret != NULL){
            if(strlen(ret) > 0){
                int allnums = Py4pdUtils_IsNumericOrDot(ret); // flag if all nums
                if(allnums){ // if errpointer is at beginning, that means we've got a float
                    double f = strtod(ret, &err);
                    SETFLOAT(&out[atompos], (t_float)f);
                }
                else{ // else we're dealing with a symbol
                    t_symbol * cursym = gensym(ret);
                    SETSYMBOL(&out[atompos], cursym);
                };
                atompos++; //increment position in atom
            };
            ret = Py4pdUtils_Mtok(NULL, sep);
        };
        if(out->a_type == A_SYMBOL){
            outlet_anything(outlet, out->a_w.w_symbol, atompos-1, out+1);
        }
        else if(out->a_type == A_FLOAT && atompos >= 1){
            outlet_list(outlet, &s_list, atompos, out);
        }
        t_freebytes(out, iptlen * sizeof(*out));
        t_freebytes(newstr, iptlen * sizeof(*newstr));
    };
    t_freebytes(sep, seplen * sizeof(*sep));
}

// =====================================================================
/*
* @brief Convert one PyObject pointer to a PureData pointer
* @param pValue is the PyObject pointer to convert
* @return the PureData pointer
*/
PyObject *Py4pdUtils_PyObjectToPointer(PyObject *pValue) {
    PyObject *pValuePointer = PyLong_FromVoidPtr(pValue);
    return pValuePointer;
}

// =====================================================================
/*
* @brief Convert one PureData pointer to a PyObject pointer
* @param p is the PureData pointer to convert
* @return the PyObject pointer
*/
PyObject *Py4pdUtils_PointerToPyObject(PyObject *p) { 
    PyObject *pValue = PyLong_AsVoidPtr(p);
    return pValue;
}

// ====================================================
/*
* @brief This function parse the arguments for the py4pd object
* @param x is the py4pd object
* @param c is the canvas of the object
* @param argc is the number of arguments
* @param argv is the arguments
* @return return void
*/

t_py *Py4pdUtils_GetObject(void){
    PyObject *pd_module = PyImport_ImportModule("pd");
    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, "py4pd");
    if (py4pd_capsule == NULL){
        return NULL;
    }
    t_py *py4pd = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
    return py4pd;
}

// =====================================================================
/*
 * @brief Run a Python function
 * @param x is the py4pd object
 * @param pArgs is the arguments to pass to the function
 * @return the return value of the function
 */
PyObject *Py4pdUtils_RunPy(t_py *x, PyObject *pArgs) { 
    t_py *prev_obj = NULL;
    int prev_obj_exists = 0;
    PyObject* MainModule = PyImport_ImportModule("pd"); 
    PyObject* oldObjectCapsule = NULL;
    PyObject* pValue = NULL;
    PyObject *objectCapsule = NULL;

    if (MainModule != NULL) {
        oldObjectCapsule = PyObject_GetAttrString(MainModule, "py4pd"); // borrowed reference
        if (oldObjectCapsule != NULL) {
            PyObject *py4pd_capsule = PyObject_GetAttrString(MainModule, "py4pd"); // borrowed reference
            prev_obj = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd"); // borrowed reference
            prev_obj_exists = 1;
            Py_DECREF(oldObjectCapsule);
            Py_DECREF(py4pd_capsule);
        }
        else{
            prev_obj_exists = 0;
            Py_XDECREF(oldObjectCapsule);
        }
    }
    else {
        pd_error(x, "[%s] Failed to import pd module when Running Python function", x->function_name->s_name);
        PyErr_Print();
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[Python] %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        PyErr_Clear();
        Py_XDECREF(MainModule);
        return NULL;
    }
    objectCapsule = Py4pdUtils_AddPdObject(x);
    if (objectCapsule == NULL){
        pd_error(x, "[Python] Failed to add object to Python");
        Py_XDECREF(MainModule);
        return NULL;
    }
    
    pValue = PyObject_CallObject(x->function, pArgs);

    t_py4pd_pValue *PyPtrValue = (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
    PyPtrValue->pValue = pValue;
    PyPtrValue->objectsUsing = 0;
    PyPtrValue->objOwner = x->objectName;

    Py4pdUtils_MemLeakCheck(pValue, 1, "Py4pd_RunPy");

    if (prev_obj_exists == 1 && pValue != NULL) {
        objectCapsule = Py4pdUtils_AddPdObject(prev_obj);
        if (objectCapsule == NULL){
            pd_error(x, "[Python] Failed to add object to Python");
        }
    }
            
    if (x->audioOutput == 1){
        return NULL;
    }

    if (pValue != NULL && x->audioOutput == 0) {
        Py4pdUtils_ConvertToPd(x, PyPtrValue, x->out1); 
        Py4pdUtils_DECREF(pValue);
        Py4pdUtils_DECREF(pArgs);
        if (pArgs->ob_refcnt != 0)
            pd_error(NULL, "[DEV] pArgs not free, Memory Leak, please report!");
        Py_XDECREF(MainModule);
        free(PyPtrValue);
        return NULL;
    }
    else if(pValue == NULL){
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[Python] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_XDECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        Py_XDECREF(pValue);
        Py4pdUtils_DECREF(pArgs);
        Py_XDECREF(MainModule);
        free(PyPtrValue);
        PyErr_Clear();
        return NULL;
    }
    else{
        Py_XDECREF(MainModule);
        Py4pdUtils_DECREF(pArgs);
        free(PyPtrValue);
        return pValue;
    }
}

// =====================================================================
/*
 * @brief Run a Python function
 * @param x is the py4pd object
 * @param pArgs is the arguments to pass to the function
 * @return the return value of the function
 */
void Py4pdUtils_INCREF(PyObject *pValue) { 
    if (pValue->ob_refcnt < 0 && PY4PD_DEBUG){
        pd_error(NULL, "[DEV] pValue refcnt < 0, Memory Leak, please report!");
        return;
    }

    if (Py_IsNone(pValue)){
        return;
    }
    // check if pValue is between -5 and 255
    else if (PyLong_Check(pValue)){
        long value = PyLong_AsLong(pValue);
        if (value >= -5 && value <= 255){
            return;
        }
    }
    Py_INCREF(pValue);
    return; 
}

// =====================================================================
/*
 * @brief Run a Python function
 * @param x is the py4pd object
 * @param pArgs is the arguments to pass to the function
 * @return the return value of the function
 */
void Py4pdUtils_DECREF(PyObject *pValue) { 
    if (pValue->ob_refcnt < 0 && PY4PD_DEBUG){
        pd_error(NULL, "[DEV] pValue refcnt < 0, Memory Leak, please report!");
        return;
    }

    if (Py_IsNone(pValue)){
        return;
    }
    // check if pValue is between -5 and 255
    else if (PyLong_Check(pValue)){
        long value = PyLong_AsLong(pValue);
        if (value >= -5 && value <= 255){
            return;
        }
    }
    if (pValue->ob_refcnt > 0){
        Py_DECREF(pValue);
    }

    
    if (pValue->ob_refcnt == 0){
        pValue = NULL;
    }
    return; 
}

// =====================================================================
/*
 * @brief Run a Python function
 * @param x is the py4pd object
 * @param pArgs is the arguments to pass to the function
 * @return the return value of the function
 */
void Py4pdUtils_KILL(PyObject *pValue) { 
    if (Py_IsNone(pValue)){
        return;
    }
    else if (Py_IsTrue(pValue)){
        return;
    }
    else if (PyLong_Check(pValue)){
        long value = PyLong_AsLong(pValue);
        if (value >= -5 && value <= 255){
            return;
        }
    }
    else{
        int iter = 0;
        while (pValue->ob_refcnt > 0){
            Py_DECREF(pValue);
            if (pValue->ob_refcnt == 0){
                pValue = NULL;
                break;
            }
            iter++;
            if (iter > 10000){
                pd_error(NULL, "[Python] Failed to kill object, there is a Memory Leak");
                break;
            }
        }
        return; 
    }
}

// =====================================================================
/*
 * @brief It warnings if there is a memory leak
 * @param pValue is the value to check
 * @return nothing
 */
void Py4pdUtils_MemLeakCheck(PyObject *pValue, int refcnt, char *where) {
    if (!PY4PD_DEBUG){
        return;
    }

    if (Py_IsNone(pValue)){
        return;
    }
    else if (Py_IsTrue(pValue)){
        return;
    }
    else if (PyLong_Check(pValue)){
        long value = PyLong_AsLong(pValue);
        if (value >= -5 && value <= 255){
            return;
        }
        else{
            if (pValue->ob_refcnt != refcnt){
                if (refcnt < pValue->ob_refcnt){
                    pd_error(NULL, "[DEV] Memory Leak inside %s, Ref Count should be %d but is %d", 
                             where, refcnt, (int)pValue->ob_refcnt);
                }
                else{
                    pd_error(NULL, "[DEV] Ref Count error in %s, Ref Count should be %d but is %d", 
                             where, refcnt, (int)pValue->ob_refcnt); 
                }
            }
        }
    }
    else{
        if (pValue->ob_refcnt != refcnt){
            if (refcnt < pValue->ob_refcnt){
                pd_error(NULL, "[DEV] Memory Leak inside %s, Ref Count should be %d but is %d", 
                         where, refcnt, (int)pValue->ob_refcnt);
            }
            else{
                pd_error(NULL, "[DEV] Ref Count error in %s, Ref Count should be %d but is %d", 
                         where, refcnt, (int)pValue->ob_refcnt); 
            }
        }
}
    return;
}

// =====================================================================
/*
* @brief Convert and output Python Values to PureData values
* @param x is the py4pd object
* @param pValue is the Python value to convert
* @return nothing, but output the value to the outlet
*/
inline void *Py4pdUtils_ConvertToPd(t_py *x, t_py4pd_pValue *pValueStruct, t_outlet *outlet) { 
    PyObject* pValue = pValueStruct->pValue;

    if (pValue->ob_refcnt < 1 && PY4PD_DEBUG){
        pd_error(NULL, "[FATAL]: When converting to pd, pValue refcnt < 1");
        return NULL;
    }
    
    if (x->outPyPointer) {
        if (pValue == Py_None && x->ignoreOnNone == 1) {
            return NULL;
        }
        t_atom pointer_atom;
        SETPOINTER(&pointer_atom, (t_gpointer *)pValueStruct);
        outlet_anything(outlet, gensym("PyObject"), 1, &pointer_atom);
        return NULL;
    }
    
    if (PyTuple_Check(pValue)){
        if (PyTuple_Size(pValue) == 1) {
            pValue = PyTuple_GetItem(pValue, 0);
        }
    }
    
    if (PyList_Check(pValue)) {  // If the function return a list list
        int list_size = PyList_Size(pValue);
        t_atom *list_array = (t_atom *)malloc(list_size * sizeof(t_atom));
        int i;
        int listIndex = 0;
        PyObject *pValue_i;
        for (i = 0; i < list_size; ++i) {
            pValue_i = PyList_GetItem(pValue, i); // borrowed reference
            if (PyLong_Check(pValue_i)) {  // If the function return a list of integers
                float result = (float)PyLong_AsLong(pValue_i); 
                list_array[listIndex].a_type = A_FLOAT;
                list_array[listIndex].a_w.w_float = result;
                listIndex++;
            } 
            else if (PyFloat_Check(pValue_i)) {  // If the function return a list of floats
                float result = PyFloat_AsDouble(pValue_i);
                list_array[listIndex].a_type = A_FLOAT;
                list_array[listIndex].a_w.w_float = result;
                listIndex++;
            } 
            else if (PyUnicode_Check(pValue_i)) {  // If the function return a
                const char *result = PyUnicode_AsUTF8(pValue_i); 
                list_array[listIndex].a_type = A_SYMBOL;
                list_array[i].a_w.w_symbol = gensym(result);
                listIndex++;

            } 
            else if (Py_IsNone(pValue_i)) {
                /* not possible to represent None in Pd, so we just skip it */
            } 
            else {
                pd_error(x,
                         "[py4pd] py4pd just convert int, float and string! "
                         "Received: %s",
                         Py_TYPE(pValue_i)->tp_name);
                // Py_DECREF(pValue);
                return 0;
            }

        }
        if (list_array[0].a_type == A_SYMBOL) {
            outlet_anything(outlet, list_array[0].a_w.w_symbol, listIndex - 1, list_array + 1);
        } 
        else {
            outlet_list(outlet, &s_list, listIndex, list_array);
        }
        free(list_array);
    } 
    else {
        if (PyLong_Check(pValue)) {
            long result = PyLong_AsLong(pValue);  // If the function return a integer
            outlet_float(outlet, result);

        } 
        else if (PyFloat_Check(pValue)) {
            double result = PyFloat_AsDouble(pValue);  // If the function return a float
            float result_float = (float)result;
            outlet_float(outlet, result_float);
        } 
        else if (PyUnicode_Check(pValue)) {
            const char *result = PyUnicode_AsUTF8(pValue); // If the function return a string
            Py4pdUtils_FromSymbolSymbol(x, gensym(result), outlet);
        } 
        else if (Py_IsNone(pValue)) {
            // Py_DECREF(pValue);
        }
        else {
            pd_error(x,
                     "[py4pd] py4pd just convert int, float and string or list "
                     "of this atoms! Received: %s",
                     Py_TYPE(pValue)->tp_name);
        }
    }
    return 0;
}

// ============================================
/*
* @brief Convert PureData Values to Python values
* @param listsArrays were the lists are stored
* @param argc is the number of arguments
* @param argv is the arguments
* @return the Python tuple with the values
*/
PyObject *Py4pdUtils_ConvertToPy(PyObject *listsArrays[], int argc, t_atom *argv) { 
    PyObject *ArgsTuple = PyTuple_New(0);  // start new tuple with 1 element
    int listStarted = 0;
    int argCount = 0;
    int listCount = 0;

    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            if (strchr(argv[i].a_w.w_symbol->s_name, '[') != NULL) {
                char *str = (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                strcpy(str, argv[i].a_w.w_symbol->s_name);
                Py4pdUtils_RemoveChar(str, '[');
                listsArrays[listCount] = PyList_New(0);
                int isNumeric = Py4pdUtils_IsNumericOrDot(str);
                if (isNumeric == 1) {
                    // check if is a float or int
                    if (strchr(str, '.') != NULL) {
                        PyList_Append(listsArrays[listCount], PyFloat_FromDouble(atof(str)));
                    } 
                    else {
                        PyList_Append(listsArrays[listCount], PyLong_FromLong(atol(str)));
                    }
                } 
                else {
                    PyList_Append(listsArrays[listCount], PyUnicode_FromString(str));
                }
                free(str);
                listStarted = 1;
            }

            else if((strchr(argv[i].a_w.w_symbol->s_name, '[') != NULL) && (strchr(argv[i].a_w.w_symbol->s_name, ']') != NULL)) {
                char *str = (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                strcpy(str, argv[i].a_w.w_symbol->s_name);
                Py4pdUtils_RemoveChar(str, '[');
                Py4pdUtils_RemoveChar(str, ']');
                listsArrays[listCount] = PyList_New(0);
                int isNumeric = Py4pdUtils_IsNumericOrDot(str);
                if (isNumeric == 1) {
                    // check if is a float or int
                    if (strchr(str, '.') != NULL) {
                        PyList_Append(listsArrays[listCount], PyFloat_FromDouble(atof(str)));
                    } 
                    else {
                        PyList_Append(listsArrays[listCount], PyLong_FromLong(atol(str)));
                    }
                } 
                else {
                    PyList_Append(listsArrays[listCount], PyUnicode_FromString(str));
                }
                free(str);
                listStarted = 1;
            }

            // ========================================
            else if (strchr(argv[i].a_w.w_symbol->s_name, ']') != NULL) {
                char *str = (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                strcpy(str, argv[i].a_w.w_symbol->s_name);
                Py4pdUtils_RemoveChar(str, ']');
                int isNumeric = Py4pdUtils_IsNumericOrDot(str);
                _PyTuple_Resize(&ArgsTuple, argCount + 1);
                if (isNumeric == 1) {
                    if (strchr(str, '.') != NULL) {
                        PyList_Append(listsArrays[listCount], PyFloat_FromDouble(atof(str)));
                    } 
                    else {
                        PyList_Append(listsArrays[listCount], PyLong_FromLong(atol(str)));
                    }
                    PyTuple_SetItem(ArgsTuple, argCount, listsArrays[listCount]);
                } 
                else {
                    PyList_Append(listsArrays[listCount], PyUnicode_FromString(str));
                    PyTuple_SetItem(ArgsTuple, argCount, listsArrays[listCount]);
                }
                // free(str);
                listStarted = 0;
                listCount++;
                argCount++;
            }

            // ========================================
            else {
                if (listStarted == 1) {
                    char *str = (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                    strcpy(str, argv[i].a_w.w_symbol->s_name);
                    int isNumeric = Py4pdUtils_IsNumericOrDot(str);
                    if (isNumeric == 1) {
                        if (strchr(str, '.') != NULL) {
                            PyList_Append(listsArrays[listCount], PyFloat_FromDouble(atof(str)));
                        } 
                        else {
                            PyList_Append(listsArrays[listCount], PyLong_FromLong(atol(str)));
                        }
                    } 
                    else {
                        PyList_Append(listsArrays[listCount], PyUnicode_FromString(str));
                    }
                    // free(str);
                } 
                else {
                    _PyTuple_Resize(&ArgsTuple, argCount + 1);
                    PyTuple_SetItem(ArgsTuple, argCount, PyUnicode_FromString(argv[i].a_w.w_symbol->s_name));
                    argCount++;
                }
            }
        } 
        else {
            if (listStarted == 1) {
                PyList_Append(listsArrays[listCount], PyFloat_FromDouble(argv[i].a_w.w_float));
            } 
            else {
                _PyTuple_Resize(&ArgsTuple, argCount + 1);
                PyTuple_SetItem(ArgsTuple, argCount, PyFloat_FromDouble(argv[i].a_w.w_float));
                argCount++;
            }
        }
    }
    return ArgsTuple;
}

// ========================= py4pd object ==============================
/*

* @brief Get the config from py4pd.cfg file
* @param x is the py4pd object
* @return the pointer to the py4pd object with the config

*/

void Py4pdUtils_SetObjConfig(t_py *x) {
    char *PADRAO_packages_path = (char *)malloc(sizeof(char) * (strlen(x->pdPatchFolder->s_name) + strlen("/py-modules") + 1));  //
    snprintf(PADRAO_packages_path, strlen(x->pdPatchFolder->s_name) + strlen("/py-modules") + 1, "%s/py-modules", x->pdPatchFolder->s_name);
    x->pkgPath = gensym(PADRAO_packages_path);
    x->runmode = 0;
    if (x->editorName == NULL){
        const char *editor = PY4PD_EDITOR;
        x->editorName = gensym(editor);
    }
    if (x->py4pdPath == NULL){
        Py4pdUtils_FindObjFolder(x);
    }

    char config_path[MAXPDSTRING];
    snprintf(config_path, sizeof(config_path), "%s/py4pd.cfg", x->py4pdPath->s_name);
    if (access(config_path, F_OK) != -1) {  // check if file exists
        FILE *file = fopen(config_path, "r");      /* should check the result */
        char line[256];                            // line buffer
        while (fgets(line, sizeof(line), file)) {  // read a line
            if (strstr(line, "packages =") != NULL) {  // check if line contains "packages ="
                char *packages_path = (char *)malloc(sizeof(char) * (strlen(line) - strlen("packages = ") + 1));  //
                strcpy(packages_path, line + strlen("packages = "));  // copy string one into the result.
                if (strlen(packages_path) > 0) {  // check if path is not empty
                    packages_path[strlen(packages_path) - 1] = '\0';  // remove the last character
                    packages_path[strlen(packages_path) - 1] = '\0';  // remove the last character
                    char *i = packages_path;
                    char *j = packages_path;
                    while (*j != 0) {
                        *i = *j++;
                        if (*i != ' ') i++;
                    }
                    *i = 0;
                    // if packages_path start with . add the home_path
                    if (packages_path[0] == '.') {
                        char *new_packages_path = (char *)malloc(sizeof(char) * (strlen(x->pdPatchFolder->s_name) + strlen(packages_path) + 1));  //
                        strcpy(new_packages_path, x->pdPatchFolder->s_name);  // copy string one into the result.
                        strcat(new_packages_path, packages_path + 1);  // append string two to the result.
                        x->pkgPath = gensym(new_packages_path);
                        free(new_packages_path);
                    } 
                    else {
                        x->pkgPath = gensym(packages_path);
                    }
                }
                free(packages_path);  // free memory
            } 
            
            else if (strstr(line, "editor =") != NULL) {
                char *editor = (char *)malloc(sizeof(char) * (strlen(line) - strlen("editor = ") + 1));  //
                strcpy(editor, line + strlen("editor = "));
                Py4pdUtils_RemoveChar(editor, '\n');
                Py4pdUtils_RemoveChar(editor, '\r');
                Py4pdUtils_RemoveChar(editor, ' ');
                x->editorName = gensym(editor);
                free(editor);  // free memory
                logpost(x, 3, "[py4pd] Editor set to %s", x->editorName->s_name);
            }
        }
        fclose(file);  // close file
    }
    free(PADRAO_packages_path);  // free memory
    Py4pdUtils_CreateTempFolder(x);
    return;
}

// ========================= PYTHON ==============================
/*
* @brief add PureData Object to Python Module
* @param x is the py4pd object
* @param capsule is the PyObject (capsule)
* @return the pointer to the py capsule

*/

PyObject *Py4pdUtils_AddPdObject(t_py *x) { 
    PyObject *MainModule = PyModule_GetDict(PyImport_AddModule("pd"));
    PyObject *objectCapsule;
    if (MainModule != NULL) {
        objectCapsule = PyDict_GetItemString(MainModule, "py4pd"); // borrowed reference
        if (objectCapsule != NULL) {
            PyCapsule_SetPointer(objectCapsule, x);
        }
        else{
            objectCapsule = PyCapsule_New(x, "py4pd", NULL);  // create a capsule to pass the object to the python interpreter
            PyModule_AddObject(PyImport_ImportModule("pd"), "py4pd", objectCapsule);  // add the capsule to the python interpreter
        }
    }
    else{
        pd_error(x, "[Python] Could not get the main module");
        objectCapsule = NULL;
    }
    return objectCapsule;
}

// ========================= GIF ==============================
/*
 * @brief This function read the gif file and return the base64 string
 * @param x is the object
 * @param filename is the gif file name
 * @return void
*/

static char* Py4pdUtils_Gif2Base64(const unsigned char* data, size_t dataSize) {
    const char base64Chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t outputSize = 4 * ((dataSize + 2) / 3);  // Calculate the output size
    char* encodedData = (char*)malloc(outputSize + 1);
    if (!encodedData) {
        printf("Memory allocation failed.\n");
        return NULL;
    }

    size_t i, j;
    for (i = 0, j = 0; i < dataSize; i += 3, j += 4) {
        unsigned char byte1 = data[i];
        unsigned char byte2 = (i + 1 < dataSize) ? data[i + 1] : 0;
        unsigned char byte3 = (i + 2 < dataSize) ? data[i + 2] : 0;

        unsigned char charIndex1 = byte1 >> 2;
        unsigned char charIndex2 = ((byte1 & 0x03) << 4) | (byte2 >> 4);
        unsigned char charIndex3 = ((byte2 & 0x0F) << 2) | (byte3 >> 6);
        unsigned char charIndex4 = byte3 & 0x3F;

        encodedData[j] = base64Chars[charIndex1];
        encodedData[j + 1] = base64Chars[charIndex2];
        encodedData[j + 2] = (i + 1 < dataSize) ? base64Chars[charIndex3] : '=';
        encodedData[j + 3] = (i + 2 < dataSize) ? base64Chars[charIndex4] : '=';
    }

    encodedData[outputSize] = '\0';  // Null-terminate the encoded data

    return encodedData;
}

// ========================= GIF ==============================
/*
 * @brief This function read the gif file and return the base64 string
 * @param x is the object
 * @param filename is the gif file name
 * @return void
*/

void Py4pdUtils_ReadGifFile(t_py *x, const char* filename){
    (void)x;
    FILE* file = fopen(filename, "rb");
    if (!file) {
        post("Unable to open file.\n");
        return;
    }

    // pixel size
    fseek(file, 6, SEEK_SET);
    fread(&x->x_width, 2, 1, file);
    fread(&x->x_height, 2, 1, file);


    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    // Allocate memory to store file contents
    unsigned char* fileContents = (unsigned char*)malloc(fileSize);
    if (!fileContents) {
        post("Memory allocation failed.\n");
        fclose(file);
        return;
    }

    // Read file contents
    size_t bytesRead = fread(fileContents, 1, fileSize, file);
    if (bytesRead != fileSize) {
        post("Failed to read file.\n");
        free(fileContents);
        fclose(file);
        return;
    }

    fclose(file);

    // Base64 encoding
    char* base64Data = Py4pdUtils_Gif2Base64(fileContents, fileSize);
    free(fileContents);

    x->x_image = base64Data;

    if (!base64Data) {
        free(base64Data);
        x->x_image = PY4PD_IMAGE;
        post("Base64 encoding failed.\n");
    }

    return;
}

// ==========================================================
// ======================= PNG ==============================
// ==========================================================
/*
 * @brief This convert get the png file and convert to base64, that can be readed for pd-gui
 * @param data is the png file
 * @param input_length is the size of the png file
 * @param encoded_data is the base64 string
 * @return void
 */

static void Py4pdUtils_Png2Base64(const uint8_t* data, size_t input_length, char* encoded_data) {
    const char base64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    size_t output_length = 4 * ((input_length + 2) / 3);
    size_t padding_length = (3 - (input_length % 3)) % 3;
    size_t encoded_length = output_length + padding_length + 1;

    for (size_t i = 0, j = 0; i < input_length;) {
        uint32_t octet_a = i < input_length ? data[i++] : 0;
        uint32_t octet_b = i < input_length ? data[i++] : 0;
        uint32_t octet_c = i < input_length ? data[i++] : 0;

        uint32_t triple = (octet_a << 0x10) + (octet_b << 0x08) + octet_c;

        encoded_data[j++] = base64_chars[(triple >> 3 * 6) & 0x3F];
        encoded_data[j++] = base64_chars[(triple >> 2 * 6) & 0x3F];
        encoded_data[j++] = base64_chars[(triple >> 1 * 6) & 0x3F];
        encoded_data[j++] = base64_chars[(triple >> 0 * 6) & 0x3F];
    }

    // Add padding if necessary
    for (size_t i = 0; i < padding_length; i++) {
        encoded_data[output_length - padding_length + i] = '=';
    }

    encoded_data[encoded_length - 1] = '\0';
}

// ===============================================
/*
 * @brief This function read the png file and convert to base64
 * @param x is the object
 * @param filename is the png file
 * @return void

*/

void Py4pdUtils_ReadPngFile(t_py *x, const char* filename){
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        pd_error(x, "Failed to open file\n");
        return;
    }

    int width, height;
    fseek(file, 16, SEEK_SET);
    fread(&width, 4, 1, file);
    fread(&height, 4, 1, file);
    width = Py4pdUtils_Ntohl(width); 
    height = Py4pdUtils_Ntohl(height);
    x->x_width = width;
    x->x_height = height;

    // Determine the file size
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    rewind(file);

    // Allocate memory for the file data
    uint8_t* file_data = (uint8_t*)malloc(file_size);
    if (file_data == NULL) {
        pd_error(x, "Failed to allocate memory\n");
        fclose(file);
        return;
    }

    // Read the file into memory
    size_t bytes_read = fread(file_data, 1, file_size, file);
    fclose(file);

    if (bytes_read != file_size) {
        pd_error(x, "Failed to read file\n");
        free(file_data);
        return ;
    }

    // Encode the file data as Base64
    size_t encoded_length = 4 * ((bytes_read + 2) / 3) + 1;
    char* base64_data = (char*)malloc(encoded_length);
    if (base64_data == NULL) {
        pd_error(x, "Failed to allocate memory\n");
        free(file_data);
        return;
    }
    Py4pdUtils_Png2Base64(file_data, bytes_read, base64_data);
    x->x_image = base64_data;
    free(file_data);
    return;
}

// ==========================================================

/* 

* @brief get the size (width and height) of the png file
* @param pngfile is the path to the png file
* @return the width and height of the png file
*
*/

inline uint32_t Py4pdUtils_Ntohl(uint32_t netlong) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return ((netlong & 0xff) << 24) |
           ((netlong & 0xff00) << 8) |
           ((netlong & 0xff0000) >> 8) |
           ((netlong & 0xff000000) >> 24);
#else
    return netlong;
#endif
}
