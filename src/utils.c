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
int parseLibraryArguments(t_py *x, PyCodeObject *code, int argc, t_atom *argv){
    int argsNumberDefined = 0;
    if (code->co_flags & CO_VARARGS) {
        x->py_arg_numbers = 1;
        int i;
        for (i = 0; i < argc; i++) {
            if (argv[i].a_type == A_SYMBOL) {
                if (strcmp(argv[i].a_w.w_symbol->s_name, "-n_args") == 0 || strcmp(argv[i].a_w.w_symbol->s_name, "-a") == 0) {
                    if (i + 1 < argc) {
                        if (argv[i + 1].a_type == A_FLOAT) {
                            x->py_arg_numbers = (int)argv[i + 1].a_w.w_float;
                            argsNumberDefined = 1;
                        }
                        else {
                            pd_error(x, "[%s] this function uses *args, you need to specify the number of arguments using -n_args (-a for short) {number}", x->objectName->s_name);
                            return 0;
                        }
                    }
                    else {
                        pd_error(x, "[%s] this function uses *args, you need to specify the number of arguments using -n_args (-a for short) {number}", x->objectName->s_name);
                        return 0;
                    }
                }
            }
        }
        if (argsNumberDefined == 0) {
            pd_error(x, "[%s] this function uses *args, you need to specify the number of arguments using -n_args (-a for short) {number}", x->objectName->s_name);
            return 0;
        }
    }
    if (code->co_flags & CO_VARKEYWORDS) {
        x->kwargs = 1;
        // pd_error(x, "[%s] function use **kwargs, **kwargs are not implemented yet", x->objectName->s_name);
        // return 0;
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

t_py *get_py4pd_object(void){
    PyObject *pd_module = PyImport_ImportModule("pd");
    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, "py4pd");
    if (py4pd_capsule == NULL){
        return NULL;
    }
    t_py *py4pd = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
    return py4pd;
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

void parsePy4pdArguments(t_py *x, t_canvas *c, int argc, t_atom *argv) {

    int i;
    for (i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            t_symbol *py4pdArgs = atom_getsymbolarg(i, argc, argv);
            if (py4pdArgs == gensym("-picture") ||
                py4pdArgs == gensym("-score") ||
                py4pdArgs == gensym("-pic") ||
                py4pdArgs == gensym("-canvas")) {
                py4pd_InitVisMode(x, c, py4pdArgs, i, argc, argv, NULL);
                x->x_outline = 1;
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
                x->out1 = outlet_new(&x->x_obj, gensym("signal"));  // create a signal outlet
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
                    &x->x_obj, gensym("signal"));  // create a signal outlet
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
        x->out1 = outlet_new(&x->x_obj, 0);  // cria um outlet caso o objeto nao contenha audio
    }
}

// ====================================================
/*
* @brief get the folder name of something
* @param x is the py4pd object
* @return save the py4pd folder in x->py4pdPath
*/
char* get_folder_name(char* path) {
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

const char* get_filename(const char* path) {
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
/*
* @brief Get the py4pd folder object, it creates the folder for scripts inside resources
* @param x is the py4pd object
* @return save the py4pd folder in x->py4pdPath
*/

void findPy4pdFolder(t_py *x){
    void* handle = dlopen(NULL, RTLD_LAZY);
    if (!handle) {
        post("Not possible to locate the folder of the py4pd object");
    }
    Dl_info info;
    if (dladdr((void*)findPy4pdFolder, &info) == 0) {
        post("Not possible to locate the folder of the py4pd object");
    }
    // remove filename from path
    #ifdef _WIN64
        // get user folder
        char *path = strdup(info.dli_fname);
        char *last_slash = strrchr(path, '\\');
        if (last_slash != NULL) {
            *last_slash = '\0';
        }
        x->py4pdPath = gensym(path);
        free(path);
    #else
        char *path = strdup(info.dli_fname);
        char *last_slash = strrchr(path, '/');
        if (last_slash != NULL) {
            *last_slash = '\0';
        }
        x->py4pdPath = gensym(path);
        free(path);
    #endif
}

// ===================================================================
/**
 * @brief Get the temp path object (inside Users/.py4pd), it creates the folder if it not exist
 * @param x is the py4pd object
 * @return save the temp path in x->tempPath
 */

void createPy4pdTempFolder(t_py *x) {
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
        }
    #else
        const char *home = getenv("HOME");
        char *temp_folder = (char *)malloc(256 * sizeof(char));
        memset(temp_folder, 0, 256);
        sprintf(temp_folder, "%s/.py4pd/", home);
        x->tempPath = gensym(temp_folder);
        if (access(temp_folder, F_OK) == -1) {
            char *command = (char *)malloc(256 * sizeof(char));
            memset(command, 0, 256);
            sprintf(command, "mkdir %s", temp_folder);
            system(command);
        }
        // free(temp_folder);

    #endif
}

// ===================================================================
/*
 * @brief It creates the commandline to open the editor
 * @param x is the py4pd object
 * @return the commandline to open the editor
 */
char *getEditorCommand(t_py *x) {
    const char *editor = x->editorName->s_name;
    const char *home = x->pdPatchFolder->s_name;
    const char *filename = x->script_name->s_name;
    char *command = (char *)malloc(256 * sizeof(char));
    memset(command, 0, 256);
    if (strcmp(editor, "vscode") == 0) {
        sprintf(command, "code '%s/%s.py'", home, filename);
    } else if (strcmp(editor, "nvim") == 0) {
        sprintf(command, "gnome-terminal -e \"nvim '%s/%s.py'\"", home,
                filename);
    } else if (strcmp(editor, "sublime") == 0) {
        sprintf(command, "subl '%s/%s.py'", home, filename);
    } else if (strcmp(editor, "emacs") == 0) {
        sprintf(command, "emacs '%s/%s.py'", home, filename);
    } else {
        pd_error(x, "[py4pd] editor %s not supported.", editor);
    }
    return command;
}

// ====================================
/*

* @brief Run command and check for errors
* @param command is the command to run
* @return void, but it prints the error if it fails

*/

void executeSystemCommand(const char *command) {
    int result = system(command);
    if (result == -1) {
        post("[py4pd] %s", command);
        return;
    }
}

// ============================================
/*

* @brief See if str is a number or a dot
* @param str is the string to check
* @return 1 if it is a number or a dot, 0 otherwise

*/

int isNumericOrDot(const char *str) {
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

void removeChar(char *str, char c) {
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
char *py4pd_mtok(char *input, char *delimiter) { 
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

void py4pd_fromsymbol_symbol(t_py *x, t_symbol *s){ 
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
        char *ret = py4pd_mtok(newstr, sep);
        char *err; // error pointer
        while(ret != NULL){
            if(strlen(ret) > 0){
                int allnums = isNumericOrDot(ret); // flag if all nums
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
            ret = py4pd_mtok(NULL, sep);
        };
        if(out->a_type == A_SYMBOL){
            outlet_anything(((t_object *)x)->ob_outlet, out->a_w.w_symbol, atompos-1, out+1);
        }
        else if(out->a_type == A_FLOAT && atompos >= 1){
            outlet_list(((t_object *)x)->ob_outlet, &s_list, atompos, out);
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
void *pyobject_to_pointer(PyObject *pValue) { 
    t_pyObjectData *data = (t_pyObjectData *)malloc(sizeof(t_pyObjectData));
    data->pValue = pValue;
    return (void *)data;
}

// =====================================================================
/*
* @brief Convert one PureData pointer to a PyObject pointer
* @param p is the PureData pointer to convert
* @return the PyObject pointer

*/

PyObject *pointer_to_pyobject(void *p) { 
    t_pyObjectData *data = (t_pyObjectData *)p;
    return data->pValue;
}

// =====================================================================
/*
* @brief Free the memory of a PyObject pointer
* @param p is the PureData pointer to free
*/

void free_pyobject_data(void *p) { 
    t_pyObjectData *data = (t_pyObjectData *)p;
    Py_XDECREF(data->pValue);
    free(data);
}

// =====================================================================
/*
* @brief Convert and output Python Values to PureData values
* @param x is the py4pd object
* @param pValue is the Python value to convert
* @return nothing, but output the value to the outlet

*/

void *py4pd_convert_to_pd(t_py *x, PyObject *pValue) { 
    if (x->outPyPointer) {
        if (pValue == Py_None && x->ignoreOnNone == 1) {
            return 0;
        }


        void *pData = pyobject_to_pointer(pValue);
        if (Py_REFCNT(pValue) == 1) {
            Py_INCREF(pValue);
        }
        t_atom pointer_atom;
        SETPOINTER(&pointer_atom, pData);
        outlet_anything(x->x_obj.ob_outlet, gensym("PyObject"), 1, &pointer_atom);
        return 0;
    }
    
    if (PyTuple_Check(pValue)){
        if (PyTuple_Size(pValue) == 1) {
            PyObject *new_pValue = PyTuple_GetItem(pValue, 0);
            Py_INCREF(new_pValue);
            Py_DECREF(pValue);
            pValue = new_pValue;
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
                float result = (float)PyLong_AsLong(pValue_i); // NOTE: Necessary to change if want double precision
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
            //  NOTE: for now, I do not  know how to represent None in Pd
            } 
            else {
                pd_error(x,
                         "[py4pd] py4pd just convert int, float and string! "
                         "Received: %s",
                         Py_TYPE(pValue_i)->tp_name);
                return 0;
            }
            // Py_DECREF(pValue_i);
        }
        if (list_array[0].a_type == A_SYMBOL) {
            outlet_anything(x->x_obj.ob_outlet, list_array[0].a_w.w_symbol, listIndex - 1, list_array + 1);
        } 
        else {
            outlet_list(x->x_obj.ob_outlet, &s_list, listIndex, list_array);
        }
        Py_DECREF(pValue);
        free(list_array);
    } 
    else {
        if (PyLong_Check(pValue)) {
            long result = PyLong_AsLong(pValue);  // If the function return a integer
            outlet_float(x->out1, result);
        } 
        else if (PyFloat_Check(pValue)) {
            double result = PyFloat_AsDouble(pValue);  // If the function return a float
            float result_float = (float)result;
            outlet_float(x->out1, result_float);
        } 
        else if (PyUnicode_Check(pValue)) {
            const char *result = PyUnicode_AsUTF8(pValue); // If the function return a string
            py4pd_fromsymbol_symbol(x, gensym(result));
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
PyObject *py4pd_convert_to_py(PyObject *listsArrays[], int argc, t_atom *argv) { 
    PyObject *ArgsTuple = PyTuple_New(0);  // start new tuple with 1 element
    int listStarted = 0;
    int argCount = 0;
    int listCount = 0;

    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            if (strchr(argv[i].a_w.w_symbol->s_name, '[') != NULL) {
                char *str = (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                strcpy(str, argv[i].a_w.w_symbol->s_name);
                removeChar(str, '[');
                listsArrays[listCount] = PyList_New(0);
                int isNumeric = isNumericOrDot(str);
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

            /*    TODO: TEST THIS
            else if((strchr(argv[i].a_w.w_symbol->s_name, '[') != NULL) && (strchr(argv[i].a_w.w_symbol->s_name, ']') != NULL)) {
                char *str = (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                strcpy(str, argv[i].a_w.w_symbol->s_name);
                removeChar(str, '[');
                removeChar(str, ']');
                listsArrays[listCount] = PyList_New(0);
                int isNumeric = isNumericOrDot(str);
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
            */

            // ========================================
            else if (strchr(argv[i].a_w.w_symbol->s_name, ']') != NULL) {
                char *str = (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                strcpy(str, argv[i].a_w.w_symbol->s_name);
                removeChar(str, ']');
                int isNumeric = isNumericOrDot(str);
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
                    int isNumeric = isNumericOrDot(str);
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

void setPy4pdConfig(t_py *x) {
    char *PADRAO_packages_path = (char *)malloc(sizeof(char) * (strlen(x->pdPatchFolder->s_name) + strlen("/py-modules/") + 1));  //
    snprintf(PADRAO_packages_path, strlen(x->pdPatchFolder->s_name) + strlen("/py-modules/") + 1, "%s/py-modules/", x->pdPatchFolder->s_name);
    x->pkgPath = gensym(PADRAO_packages_path);
    x->runmode = 0;
    if (x->editorName == NULL){
        const char *editor = PY4PD_EDITOR;
        x->editorName = gensym(editor);
    }
    char config_path[PATH_MAX];
    snprintf(config_path, sizeof(config_path), "%s/py4pd.cfg", x->pdPatchFolder->s_name);
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
            else if (strstr(line, "thread =") != NULL) {
                char *thread = (char *)malloc(sizeof(char) * (strlen(line) - strlen("thread = ") + 1));  //
                strcpy(thread, line + strlen("thread = ")); 
                if (strlen(thread) > 0) {  
                    thread[strlen(thread) - 1] = '\0';  
                    thread[strlen(thread) - 1] = '\0'; 
                    char *i = thread;
                    char *j = thread;
                    while (*j != 0) {
                        *i = *j++;
                        if (*i != ' ') i++;
                    }
                    *i = 0;
                    if (thread[0] == '1') {
                        x->runmode = 1;
                    } 
                    else {
                        x->runmode = 0;
                    }
                }
                free(thread);  // free memory
            } 
            else if (strstr(line, "editor =") != NULL) {
                char *editor = (char *)malloc(sizeof(char) * (strlen(line) - strlen("editor = ") + 1));  //
                strcpy(editor, line + strlen("editor = "));
                removeChar(editor, '\n');
                removeChar(editor, '\r');
                removeChar(editor, ' ');
                x->editorName = gensym(editor);
                free(editor);  // free memory
            }
        }
        fclose(file);  // close file
    }

    free(PADRAO_packages_path);  // free memory
    return;
}

// ========================= PYTHON ==============================

/*

* @brief add PureData Object to Python Module
* @param x is the py4pd object
* @param capsule is the PyObject (capsule)
* @return the pointer to the py capsule

*/

PyObject *py4pd_add_pd_object(t_py *x) { 
    PyObject *MainModule = PyModule_GetDict(PyImport_AddModule("pd"));
    PyObject *objectCapsule;
    if (MainModule != NULL) {
        objectCapsule = PyDict_GetItemString(MainModule, "py4pd"); // borrowed reference
        if (objectCapsule != NULL) {
            PyCapsule_SetPointer(objectCapsule, x);
        }
        else{
            objectCapsule = PyCapsule_New(x, "py4pd", NULL);  // create a capsule to pass the object to the python interpreter
            PyModule_AddObject(PyImport_AddModule("pd"), "py4pd", objectCapsule);  // add the capsule to the python interpreter
        }
    }
    else{
        pd_error(x, "[Python] Could not get the main module");
        objectCapsule = NULL;
    }
    return objectCapsule;
}

// ========================= GIF ==============================
#include <stdio.h>
#include <stdlib.h>

static char* gif2base64(const unsigned char* data, size_t dataSize) {
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
void readGifFile(t_py *x, const char* filename){
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
    char* base64Data = gif2base64(fileContents, fileSize);
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

void png2base64(const uint8_t* data, size_t input_length, char* encoded_data) {
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
void readPngFile(t_py *x, const char* filename){
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        pd_error(x, "Failed to open file\n");
        return;
    }

    int width, height;
    fseek(file, 16, SEEK_SET);
    fread(&width, 4, 1, file);
    fread(&height, 4, 1, file);
    width = py4pd_ntohl(width); 
    height = py4pd_ntohl(height);
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
    png2base64(file_data, bytes_read, base64_data);
    x->x_image = base64_data;
    free(file_data);
    return;
}




// ========================= PIP ==============================
/*

* @brief install a python package
* @param x is the py4pd object
* @param package is the name of the package
* @return 0 if success, 1 if error

*/

// ========================= PNG ==============================

/* 

* @brief get the size of a png file
* @param pngfile is the path to the png file
* @return the size of the png file

*/

uint32_t py4pd_ntohl(uint32_t netlong){ 
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return ((netlong & 0xff) << 24) |
           ((netlong & 0xff00) << 8) |
           ((netlong & 0xff0000) >> 8) |
           ((netlong & 0xff000000) >> 24);
#else
    return netlong;
#endif
}

