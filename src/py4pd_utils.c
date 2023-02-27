#include "pd_module.h"


// ===================================================================
// ======================== Work with lists in PureData ==============
// ===================================================================

// create a hash table of PyObjects lists, where the key is the name of the list and when I add a new value,
// I just append to the list. When I want to get the list, I just get the list from the hash table
// This hash table will be responsable to create a new list if the list does not exist
// and to append a new value to the list if the list already exists

// ===================================================================
char* get_editor_command(t_py *x) {
    const char* editor = x->editorName->s_name;
    const char* home = x->home_path->s_name;
    const char* filename = x->script_name->s_name;
    char* command = (char*)malloc(256 * sizeof(char));
    memset(command, 0, 256);
    if (strcmp(editor, "vscode") == 0) {
        sprintf(command, "code '%s/%s.py'", home, filename);
    } else if (strcmp(editor, "nvim") == 0) {
        sprintf(command, "gnome-terminal -e \"nvim '%s/%s.py'\"", home, filename);
    } else if (strcmp(editor, "sublime") == 0) {
        sprintf(command, "subl '%s/%s.py'", home, filename);
    } else if (strcmp(editor, "emacs") == 0) {
        sprintf(command, "emacs '%s/%s.py'", home, filename);
    } else {
        pd_error(x, "[py4pd] editor %s not supported.", editor);
    }
    return command;
}


// ============================================
int isNumericOrDot(const char *str) {
    int hasDot = 0;
    while (*str) {
        if (isdigit(*str)) {
            str++;
        }
        else if (*str == '.' && !hasDot) {
            hasDot = 0;
            str++;
        }
        else {
            return 0;
        }
    }
    return 1;
}

// =====================================================================

void removeChar(char* str, char c) {
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

void *py4pd_convert_to_pd(t_py *x, PyObject *pValue) {
    
    if (PyList_Check(pValue)){                       // If the function return a list list
        int list_size = PyList_Size(pValue);
        t_atom *list_array = (t_atom *) malloc(list_size * sizeof(t_atom));     
        int i;       
        for (i = 0; i < list_size; ++i) {
            PyObject *pValue_i = PyList_GetItem(pValue, i);
            if (PyLong_Check(pValue_i)) {            // If the function return a list of integers
                long result = PyLong_AsLong(pValue_i);
                float result_float = (float)result;
                list_array[i].a_type = A_FLOAT;
                list_array[i].a_w.w_float = result_float;

            } 
            else if (PyFloat_Check(pValue_i)) {    // If the function return a list of floats
                double result = PyFloat_AsDouble(pValue_i);
                float result_float = (float)result;
                list_array[i].a_type = A_FLOAT;
                list_array[i].a_w.w_float = result_float;
            } 
            else if (PyUnicode_Check(pValue_i)) {  // If the function return a list of strings
                const char *result = PyUnicode_AsUTF8(pValue_i); 
                list_array[i].a_type = A_SYMBOL;
                list_array[i].a_w.w_symbol = gensym(result);
            } 
            else if (Py_IsNone(pValue_i)) {        // If the function return a list of None
            } 
            else {
                pd_error(x, "[py4pd] py4pd just convert int, float and string! Received: %s", Py_TYPE(pValue_i)->tp_name);
                Py_DECREF(pValue_i);
                return 0;
            }
        }
        outlet_list(x->out_A, 0, list_size, list_array); 
        free(list_array);
    } 
    else {
        if (PyLong_Check(pValue)) {
            long result = PyLong_AsLong(pValue); // If the function return a integer
            outlet_float(x->out_A, result);
        } 
        else if (PyFloat_Check(pValue)) {
            double result = PyFloat_AsDouble(pValue); // If the function return a float
            float result_float = (float)result;
            outlet_float(x->out_A, result_float);
        }
        else if (PyUnicode_Check(pValue)) {
            const char *result = PyUnicode_AsUTF8(pValue); // If the function return a string
            outlet_symbol(x->out_A, gensym(result)); 
        } 
        else if (Py_IsNone(pValue)) {
            // post("[py4pd] function %s return None", x->function_name->s_name); // TODO: Thing about this
        } 
        // when function not use return    
        else {
            pd_error(x, "[py4pd] py4pd just convert int, float and string or list of this atoms! Received: %s", Py_TYPE(pValue)->tp_name);
        }
        





    //     } else if (Py_IsNone(pValue)) {
    //         outlet_bang(x->out_A); // If the function return a None
    //     } else {
    //         pd_error(x, "[py4pd] py4pd just convert int, float and string or list of this atoms! Received: %s", Py_TYPE(pValue)->tp_name);
    //     }
    //
    // 



    }
    return 0;

}

// ============================================
// create an function that with input ArgsTuple, listsArrays, argc, argv, x

void *py4pd_convert_to_py(PyObject *listsArrays[], int argc, t_atom *argv) {
    PyObject *ArgsTuple = PyTuple_New(0); // start new tuple with 1 element
    int listStarted = 0;
    int argCount = 0;
    int listCount = 0;

    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            //   TODO: Create way to work with things like [1], [a], [casa] | One thing in list

            // ========================================
            if (strchr(argv[i].a_w.w_symbol->s_name, '[') != NULL){ 
                char *str = (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                strcpy(str, argv[i].a_w.w_symbol->s_name);
                removeChar(str, '[');
                listsArrays[listCount] = PyList_New(0);
                int isNumeric = isNumericOrDot(str);
                if (isNumeric == 1){
                    PyList_Append(listsArrays[listCount], PyFloat_FromDouble(atof(str)));
                }
                else {
                    PyList_Append(listsArrays[listCount], PyUnicode_FromString(str));
                }
                listStarted = 1;
            }

            // ========================================
            else if (strchr(argv[i].a_w.w_symbol->s_name, ']') != NULL){
                char *str = (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                strcpy(str, argv[i].a_w.w_symbol->s_name);
                removeChar(str, ']');
                int isNumeric = isNumericOrDot(str);
                _PyTuple_Resize(&ArgsTuple, argCount + 1);
                if (isNumeric == 1){
                    PyList_Append(listsArrays[listCount], PyFloat_FromDouble(atof(str)));
                    PyTuple_SetItem(ArgsTuple, argCount, listsArrays[listCount]);
                }
                else {
                    PyList_Append(listsArrays[listCount], PyUnicode_FromString(str));
                    PyTuple_SetItem(ArgsTuple, argCount, listsArrays[listCount]);
                }
                listStarted = 0;
                listCount++;
                argCount++;
            }

            // ========================================
            else {
                if (listStarted == 1){
                    char *str = (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                    strcpy(str, argv[i].a_w.w_symbol->s_name);
                    int isNumeric = isNumericOrDot(str);
                    if (isNumeric == 1){
                        PyList_Append(listsArrays[listCount], PyFloat_FromDouble(atof(str)));
                    }
                    else {
                        PyList_Append(listsArrays[listCount], PyUnicode_FromString(str));
                    }
                }
                else {
                    _PyTuple_Resize(&ArgsTuple, argCount + 1);
                    PyTuple_SetItem(ArgsTuple, argCount, PyUnicode_FromString(argv[i].a_w.w_symbol->s_name));
                    argCount++;
                }
            }
        }
        else{
            if (listStarted == 1){
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

int *set_py4pd_config(t_py *x) {
    x->packages_path = gensym("./py-modules/");
    x->thread = 2;
    x->editorName = gensym("vscode");
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
                    packages_path[strlen(packages_path) - 1] = '\0'; // remove the last character
                    packages_path[strlen(packages_path) - 1] = '\0'; // remove the last character
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
                        x->thread = 1;
                    } else {
                        x->thread = 2;
                    }

                }
            }
            else if (strstr(line, "editor =") != NULL){
                // get editor name
                char *editor = (char *)malloc(sizeof(char) * (strlen(line) - strlen("editor = ") + 1)); //
                strcpy(editor, line + strlen("editor = ")); 
                removeChar(editor, '\n');
                removeChar(editor, '\r');
                removeChar(editor, ' ');
                x->editorName = gensym(editor);
                post("[py4pd] Editor set to %s", x->editorName->s_name);
            }
        }
        fclose(file); // close file
    } 

    return 0;
}


