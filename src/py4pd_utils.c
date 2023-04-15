#include "m_pd.h"
#include "pd_module.h"
#include "py4pd.h"
#include "py4pd_utils.h"
#include "tupleobject.h"


// ====================================================
/**
* @brief Get the py4pd folder object, it creates the folder for scripts inside resources
 * @param x is the py4pd object
 * @return save the py4pd folder in x->py4pd_folder
 */

void findpy4pd_folder(t_py *x){
    void* handle = dlopen(NULL, RTLD_LAZY);
    if (!handle) {
        post("Not possible to locate the folder of the py4pd object");
    }
    Dl_info info;
    if (dladdr((void*)findpy4pd_folder, &info) == 0) {
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
        x->py4pd_folder = gensym(path);
        free(path);
        const char *py4pd_folder = x->py4pd_folder->s_name;
        LPSTR py4pdScripts = (LPSTR)malloc(256 * sizeof(char));
        memset(py4pdScripts, 0, 256);
        sprintf(py4pdScripts, "%s\\resources\\scripts", py4pd_folder);
        x->py4pd_scripts = gensym(py4pdScripts);
        if (access(py4pdScripts, F_OK) == -1) {
            char *command = (char *)malloc(256 * sizeof(char));
            memset(command, 0, 256);
            if (!CreateDirectory(py4pdScripts, NULL)){
                post("Failed to create directory, Report, this create instabilities: %d\n", GetLastError());
            }
        }
    #else
        char *path = strdup(info.dli_fname);
        char *last_slash = strrchr(path, '/');
        if (last_slash != NULL) {
            *last_slash = '\0';
        }
        x->py4pd_folder = gensym(path);
        free(path);
        const char *py4pd_folder = x->py4pd_folder->s_name;
        char *py4pdScripts = (char *)malloc(256 * sizeof(char));
        memset(py4pdScripts, 0, 256);
        sprintf(py4pdScripts, "%s/resources/scripts", py4pd_folder);
        x->py4pd_scripts = gensym(py4pdScripts);
        if (access(py4pdScripts, F_OK) == -1) {
            char *command = (char *)malloc(256 * sizeof(char));
            memset(command, 0, 256);
            sprintf(command, "mkdir %s", py4pdScripts);
            system(command);
        }
        // free(py4pdScripts);
    #endif
}

// ===================================================================
/**
 * @brief Get the temp path object (inside Users/.py4pd), it creates the folder if it not exist
 * * @param x is the py4pd object
 * @return save the temp path in x->temp_folder
 */

void py4pd_tempfolder(t_py *x) {
    #ifdef _WIN64
        // get user folder
        char *user_folder = getenv("USERPROFILE");
        LPSTR home = (LPSTR)malloc(256 * sizeof(char));
        memset(home, 0, 256);
        sprintf(home, "%s\\.py4pd\\", user_folder);
        x->temp_folder = gensym(home);
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
        x->temp_folder = gensym(temp_folder);
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
/**
 * @brief It creates the commandline to open the editor
 * @param x is the py4pd object
 * @return the commandline to open the editor
 */
char *get_editor_command(t_py *x) {
    const char *editor = x->editorName->s_name;
    const char *home = x->home_path->s_name;
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
/**
 * @brief Run command and check for errors
 * @param command is the command to run
 * @return void, but it prints the error if it fails
 */

void pd4py_system_func(const char *command) {
    int result = system(command);
    if (result == -1) {
        post("[py4pd] %s", command);
        return;
    }
}

// ============================================
/**
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
/**
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
    // adapted from stack overflow - Derek Kwan
    // designed to work like strtok
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
/**
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
        // get length of input string
        long unsigned int iptlen = strlen(s->s_name);
        // allocate t_atom [] on length of string
        // hacky way of making sure there's enough space
        t_atom* out = t_getbytes(iptlen * sizeof(*out));
        iptlen++;
        char *newstr = t_getbytes(iptlen * sizeof(*newstr));
        memset(newstr, '\0', iptlen);
        strcpy(newstr, s->s_name);
        int atompos = 0; //position in atom
        // parsing by token
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
void *pyobject_to_pointer(PyObject *pValue) {
    t_pyObjectData *data = (t_pyObjectData *)malloc(sizeof(t_pyObjectData));
    data->pValue = pValue;
    return (void *)data;
}

// =====================================================================
PyObject *pointer_to_pyobject(void *p) {
    t_pyObjectData *data = (t_pyObjectData *)p;
    return data->pValue;
}

// =====================================================================
void free_pyobject_data(void *p) {
    t_pyObjectData *data = (t_pyObjectData *)p;
    Py_XDECREF(data->pValue);
    free(data);
}

// =====================================================================
/**
 * @brief Convert and output Python Values to PureData values
 * @param x is the py4pd object
 * @param pValue is the Python value to convert
 * @return nothing, but output the value to the outlet
 */

void *py4pd_convert_to_pd(t_py *x, PyObject *pValue) { // TODO: fix the type of the output
    if (x->outPyPointer) {
        void *pData = pyobject_to_pointer(pValue);
        if (Py_REFCNT(pValue) == 1) {
            Py_INCREF(pValue);
        }
        t_atom pointer_atom;
        SETPOINTER(&pointer_atom, pData);
        outlet_anything(x->x_obj.ob_outlet, gensym("PyObject"), 1, &pointer_atom);
        return 0;
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
            outlet_float(x->out_A, result);
        } 
        else if (PyFloat_Check(pValue)) {
            double result = PyFloat_AsDouble(pValue);  // If the function return a float
            float result_float = (float)result;
            outlet_float(x->out_A, result_float);
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
/**
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
            //   TODO: Create way to work with things like [1], [a], [casa] |
            // ========================================
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
                // free(str);
                listStarted = 1;
            }

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
/**
 * @brief Get the config from py4pd.cfg file
 * @param x is the py4pd object
 * @return the pointer to the py4pd object with the config
 */

void set_py4pd_config(t_py *x) {
    char *PADRAO_packages_path = (char *)malloc(sizeof(char) * (strlen(x->home_path->s_name) + strlen("/py-modules/") + 1));  //
    snprintf(PADRAO_packages_path, strlen(x->home_path->s_name) + strlen("/py-modules/") + 1, "%s/py-modules/", x->home_path->s_name);
    x->packages_path = gensym(PADRAO_packages_path);
    x->runmode = 0;
    if (x->editorName == NULL){
        const char *editor = PY4PD_EDITOR;
        x->editorName = gensym(editor);
    }
    char config_path[PATH_MAX];
    snprintf(config_path, sizeof(config_path), "%s/py4pd.cfg", x->home_path->s_name);
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
                        char *new_packages_path = (char *)malloc(sizeof(char) * (strlen(x->home_path->s_name) + strlen(packages_path) + 1));  //
                        strcpy(new_packages_path, x->home_path->s_name);  // copy string one into the result.
                        strcat(new_packages_path, packages_path + 1);  // append string two to the result.
                        x->packages_path = gensym(new_packages_path);
                        free(new_packages_path);
                    } 
                    else {
                        x->packages_path = gensym(packages_path);
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

/**
 * @brief add PureData Object to Python Module
 * @param x is the py4pd object
 * @param capsule is the PyObject (capsule)
 * @return the pointer to the py capsule
 */

PyObject *py4pd_add_pd_object(t_py *x) {
    PyObject *MainModule = PyModule_GetDict(PyImport_AddModule("__main__"));
    PyObject *objectCapsule;
    if (MainModule != NULL) {
        objectCapsule = PyDict_GetItemString(MainModule, "py4pd"); // borrowed reference
        if (objectCapsule != NULL) {
            PyCapsule_SetPointer(objectCapsule, x);
        }
        else{
            objectCapsule = PyCapsule_New(x, "py4pd", NULL);  // create a capsule to pass the object to the python interpreter
            PyModule_AddObject(PyImport_AddModule("__main__"), "py4pd", objectCapsule);  // add the capsule to the python interpreter
        }
    }
    else{
        pd_error(x, "[Python] Could not get the main module");
        objectCapsule = NULL;
    }
    return objectCapsule;
}

// ========================= PNG ==============================

// ntohl is one function that is not available on Windows, so we need to define it

uint32_t py4pd_ntohl(uint32_t netlong){ // ntohl exists on windows but not on mac
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return ((netlong & 0xff) << 24) |
           ((netlong & 0xff00) << 8) |
           ((netlong & 0xff0000) >> 8) |
           ((netlong & 0xff000000) >> 24);
#else
    return netlong;
#endif
}


//
// int get_png_size(const char *pngfile){
//     FILE *file = fopen(pngfile, "rb");
//
//     uint8_t header[8];
//     if (fread(header, 1, sizeof(header), file) != sizeof(header)) {
//         printf("Failed to read PNG header\n");
//         fclose(file);
//         return 1;
//     }
//
//     if (png_sig_cmp(header, 0, sizeof(header)) != 0) {
//         printf("File is not a PNG\n");
//         fclose(file);
//         return 1;
//     }
//
//     uint32_t length;
//     if (fread(&length, sizeof(length), 1, file) != 1) {
//         printf("Failed to read chunk length\n");
//         fclose(file);
//         return 1;
//     }
//     length = ntohl(length);
//
//     uint8_t chunk_type[5];
//     if (fread(chunk_type, 1, sizeof(chunk_type), file) != sizeof(chunk_type)) {
//         printf("Failed to read chunk type\n");
//         fclose(file);
//         return 1;
//     }
//     chunk_type[sizeof(chunk_type) - 1] = '\0';
//
//     if (strcmp((char*)chunk_type, "IHDR") != 0) {
//         printf("First chunk is not IHDR\n");
//         fclose(file);
//         return 1;
//     }
//
//     uint32_t width, height;
//     if (fread(&width, sizeof(width), 1, file) != 1 ||
//         fread(&height, sizeof(height), 1, file) != 1) {
//         printf("Failed to read PNG dimensions\n");
//         fclose(file);
//         return 1;
//     }
//     width = ntohl(width);
//     height = ntohl(height);
//
//     post("Width: %u, Height: %u\n", width, height);
//
//     fclose(file);
//
//     return 0;
// }
