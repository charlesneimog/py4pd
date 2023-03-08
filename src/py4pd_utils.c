#include "m_pd.h"
#include "pd_module.h"
#include "py4pd_utils.h"


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
    char *path = strdup(info.dli_fname);
    char *last_slash = strrchr(path, '/');
    if (last_slash != NULL) {
        *last_slash = '\0';
    }
    x->py4pd_folder = gensym(path);
    #ifdef _WIN64
        // get user folder
        char *py4pd_folder = x->py4pd_folder->s_name;
        // create folder x->py4pd_folder/resources/scripts
        LPSTR py4pdScripts = (LPSTR)malloc(256 * sizeof(char));
        memset(py4pdScripts, 0, 256);
        sprintf(py4pdScripts, "%s\\resources\\scripts\\", py4pd_folder);
        x->py4pd_scripts = gensym(py4pdScripts);
        if (!CreateDirectory(py4pdScripts, NULL)){
            post("Failed to create directory, Report, this create instabilities: %d\n", GetLastError());
        }
    #else
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
        if (!CreateDirectory(home, NULL)){
            post("Failed to create directory: %d\n", GetLastError());
        }
        if (!SetFileAttributes(home, FILE_ATTRIBUTE_HIDDEN)){
            post("Failed to set hidden attribute: %d\n", GetLastError());
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
/**
 * @brief Convert and output Python Values to PureData values
 * @param x is the py4pd object
 * @param pValue is the Python value to convert
 * @return nothing, but output the value to the outlet
 */

void *py4pd_convert_to_pd(t_py *x, PyObject *pValue) {
    if (PyList_Check(pValue)) {  // If the function return a list list
        int list_size = PyList_Size(pValue);
        t_atom *list_array = (t_atom *)malloc(list_size * sizeof(t_atom));
        int i;
        int listIndex = 0;
        for (i = 0; i < list_size; ++i) {
            PyObject *pValue_i = PyList_GetItem(pValue, i);
            if (PyLong_Check(
                    pValue_i)) {  // If the function return a list of integers
                long result = PyLong_AsLong(pValue_i);
                float result_float = (float)result;
                list_array[listIndex].a_type = A_FLOAT;
                list_array[listIndex].a_w.w_float = result_float;
                listIndex++;

            } else if (PyFloat_Check(pValue_i)) {  // If the function return a
                                                   // list of floats
                double result = PyFloat_AsDouble(pValue_i);
                float result_float = (float)result;
                list_array[listIndex].a_type = A_FLOAT;
                list_array[listIndex].a_w.w_float = result_float;
                listIndex++;
            } else if (PyUnicode_Check(pValue_i)) {  // If the function return a
                                                     // list of strings
                const char *result = PyUnicode_AsUTF8(pValue_i);
                if (strchr(result, ' ') != NULL) {
                    char *result_copy = (char *)malloc(strlen(result) * sizeof(char));
                    strcpy(result_copy, result);
                    char *token = strtok(result_copy, " ");
                    while (token != NULL) {
                        list_array = (t_atom *)realloc(list_array, (list_size + 1) * sizeof(t_atom));
                        list_array[listIndex].a_type = A_GIMME; // A_SYMBOL; 
                        list_array[listIndex].a_w.w_symbol = gensym(token);
                        listIndex++;
                        list_size++;
                        token = strtok(NULL, " ");
                    }
                    free(result_copy);
                    continue;
                }
            } else if (Py_IsNone(pValue_i)) {  // If the function return a list
                                               // of None
            } else {
                pd_error(x,
                         "[py4pd] py4pd just convert int, float and string! "
                         "Received: %s",
                         Py_TYPE(pValue_i)->tp_name);
                Py_DECREF(pValue_i);
                return 0;
            }
        }
        outlet_list(x->out_A, 0, list_size, list_array);
        free(list_array);
    } else {
        if (PyLong_Check(pValue)) {
            long result =
                PyLong_AsLong(pValue);  // If the function return a integer
            outlet_float(x->out_A, result);
        } else if (PyFloat_Check(pValue)) {
            double result = PyFloat_AsDouble(pValue);  // If the function return a float
            float result_float = (float)result;
            outlet_float(x->out_A, result_float);
        } else if (PyUnicode_Check(pValue)) {
            const char *result = PyUnicode_AsUTF8(pValue);  // If the function return a string
            // check if there is a space
            char *result_copy = (char *)malloc(strlen(result) * sizeof(char));   
            strcpy(result_copy, result);
            char *token = strtok(result_copy, " ");
            if (token != NULL) {
                t_atom *list_array = (t_atom *)malloc(sizeof(t_atom));
                int list_size = 0;
                int listIndex = 0;
                while (token != NULL) {
                    list_array = (t_atom *)realloc(list_array, (list_size + 1) * sizeof(t_atom));
                    list_array[listIndex].a_type = A_GIMME; // A_SYMBOL; 
                    list_array[listIndex].a_w.w_symbol = gensym(token);
                    listIndex++;
                    list_size++;
                    token = strtok(NULL, " ");
                }
                outlet_list(x->out_A, 0, list_size, list_array);
                free(list_array);
                free(result_copy);
                return 0;
            }

        } else if (Py_IsNone(pValue)) {
            // x->function_name->s_name); // TODO: Thing about this
        }
        // when function not use return
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
void *py4pd_convert_to_py(PyObject *listsArrays[], int argc, t_atom *argv) {
    PyObject *ArgsTuple = PyTuple_New(0);  // start new tuple with 1 element
    int listStarted = 0;
    int argCount = 0;
    int listCount = 0;

    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            //   TODO: Create way to work with things like [1], [a], [casa] |
            //   One thing in list

            // ========================================
            if (strchr(argv[i].a_w.w_symbol->s_name, '[') != NULL) {
                char *str =
                    (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                strcpy(str, argv[i].a_w.w_symbol->s_name);
                removeChar(str, '[');
                listsArrays[listCount] = PyList_New(0);
                int isNumeric = isNumericOrDot(str);
                if (isNumeric == 1) {
                    PyList_Append(listsArrays[listCount],
                                  PyFloat_FromDouble(atof(str)));
                } else {
                    PyList_Append(listsArrays[listCount],
                                  PyUnicode_FromString(str));
                }
                listStarted = 1;
            }

            // ========================================
            else if (strchr(argv[i].a_w.w_symbol->s_name, ']') != NULL) {
                char *str =
                    (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                strcpy(str, argv[i].a_w.w_symbol->s_name);
                removeChar(str, ']');
                int isNumeric = isNumericOrDot(str);
                _PyTuple_Resize(&ArgsTuple, argCount + 1);
                if (isNumeric == 1) {
                    PyList_Append(listsArrays[listCount],
                                  PyFloat_FromDouble(atof(str)));
                    PyTuple_SetItem(ArgsTuple, argCount,
                                    listsArrays[listCount]);
                } else {
                    PyList_Append(listsArrays[listCount],
                                  PyUnicode_FromString(str));
                    PyTuple_SetItem(ArgsTuple, argCount,
                                    listsArrays[listCount]);
                }
                listStarted = 0;
                listCount++;
                argCount++;
            }

            // ========================================
            else {
                if (listStarted == 1) {
                    char *str = (char *)malloc(
                        strlen(argv[i].a_w.w_symbol->s_name) + 1);
                    strcpy(str, argv[i].a_w.w_symbol->s_name);
                    int isNumeric = isNumericOrDot(str);
                    if (isNumeric == 1) {
                        PyList_Append(listsArrays[listCount],
                                      PyFloat_FromDouble(atof(str)));
                    } else {
                        PyList_Append(listsArrays[listCount],
                                      PyUnicode_FromString(str));
                    }
                } else {
                    _PyTuple_Resize(&ArgsTuple, argCount + 1);
                    PyTuple_SetItem(
                        ArgsTuple, argCount,
                        PyUnicode_FromString(argv[i].a_w.w_symbol->s_name));
                    argCount++;
                }
            }
        } else {
            if (listStarted == 1) {
                PyList_Append(listsArrays[listCount],
                              PyFloat_FromDouble(argv[i].a_w.w_float));
            } else {
                _PyTuple_Resize(&ArgsTuple, argCount + 1);
                PyTuple_SetItem(ArgsTuple, argCount,
                                PyFloat_FromDouble(argv[i].a_w.w_float));
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

int *set_py4pd_config(t_py *x) {
    char *PADRAO_packages_path = (char *)malloc(sizeof(char) * (strlen(x->home_path->s_name) + strlen("/py-modules/") + 1));  //
    strcpy(PADRAO_packages_path, x->home_path->s_name);  // copy string one into the result.
    strcat(PADRAO_packages_path, "/py-modules/");  // append string two to the result.
    x->packages_path = gensym(PADRAO_packages_path);
    x->thread = 0;
    x->editorName = gensym("vscode");
    char *config_path = (char *)malloc(sizeof(char) * (strlen(x->home_path->s_name) + strlen("/py4pd.cfg") + 1));  //
    strcpy(config_path, x->home_path->s_name);           // copy string one into the result.
    strcat(config_path, "/py4pd.cfg");      // append string two to the result.
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
                    } else {
                        x->packages_path = gensym(packages_path);
                    }
                }
                free(packages_path);  // free memory
            } else if (strstr(line, "thread =") != NULL) {
                char *thread = (char *)malloc(
                    sizeof(char) *
                    (strlen(line) - strlen("thread = ") + 1));  //
                strcpy(
                    thread,
                    line +
                        strlen("thread = "));  // copy string one into the
                                               // result. TODO: implement thread
                // print thread value
                if (strlen(thread) > 0) {  // check if path is not empty
                    // from thread remove the two last character
                    thread[strlen(thread) - 1] = '\0';  // remove the last character
                    thread[strlen(thread) - 1] = '\0';  // remove the last character

                    char *i = thread;
                    char *j = thread;
                    while (*j != 0) {
                        *i = *j++;
                        if (*i != ' ') i++;
                    }
                    *i = 0;
                    // if thread start with . add the home_path
                    if (thread[0] == '1') {
                        x->thread = 1;
                    } else {
                        x->thread = 0;
                    }
                }
            } else if (strstr(line, "editor =") != NULL) {
                // get editor name
                char *editor = (char *)malloc(
                    sizeof(char) *
                    (strlen(line) - strlen("editor = ") + 1));  //
                strcpy(editor, line + strlen("editor = "));
                removeChar(editor, '\n');
                removeChar(editor, '\r');
                removeChar(editor, ' ');
                x->editorName = gensym(editor);
                post("[py4pd] Editor set to %s", x->editorName->s_name);
            }
        }
        fclose(file);  // close file
    }

    return 0;
}


