#ifndef PY4PD_H
#define PY4PD_H
#include <m_pd.h>
#include <pthread.h>
#include <regex.h> // for regex 

#ifdef _WIN64 // If windows 64bits include 
    #include <windows.h>
#endif

#ifdef HAVE_UNISTD_H // from Pure Data source code
    #include <unistd.h>
#endif

// Python include
#define PY_SSIZE_T_CLEAN // Good practice to use this
#include <Python.h>

/* ========= MAIN LIST ========= 

    TODO: Way to set global variables, I think that will be important for things like general path (lilypond, etc)
        
    TODO: If the run method set before the end of the thread, there is an error, that close all PureData.

    TODO: The set function need to run after the load of the object, but before the start of the thread???
        It does not load external modules in the pd_new.

    TODO: system("nvim -c ':terminal' /path/to/file"); // open a terminal in the file -- NVIM
       // system("code /path/to/file"); // open a terminal in the file -- VSCode 
       // system("subl /path/to/file"); // open a terminal in the file -- Sublime Text
        

*/

// =====================================
typedef struct _py { // It seems that all the objects are some kind of class.
    t_object            x_obj; // convensao no puredata source code
    t_canvas            *x_canvas; // pointer to the canvas
    t_int               object_number; // object number
    t_inlet             *in1;
    PyObject            *module; // python object
    PyObject            *function; // function name 
    t_int               thread; // arguments
    t_int               function_called; // flag to check if the set function was called
    t_int               py_arg_numbers; // number of arguments
    t_symbol            *editorName; // editor name
    t_symbol            *packages_path; // packages path 
    t_symbol            *home_path; // home path this always is the path folder (?)
    t_symbol            *object_path; // save object path TODO: want to save scripts inside this folder and make all accessible
    t_symbol            *function_name; // function name
    t_symbol            *script_name; // script name
    t_outlet            *out_A; // outlet 1.
}t_py;

// create a pointer for the t_py class

static t_class *py4pd_class;
static int object_count = 1;
static t_py *py4pd_object;

// create and array of pointers for the t_py class
static t_py *py4pd_object_array[100];

// Set Debug mode
#define DEBUG 1

#endif
