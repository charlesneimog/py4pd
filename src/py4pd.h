// =================================
// https://github.com/pure-data/pure-data/src/x_gui.c 

// if py4pd is not defined, define it
#ifndef PY4PD_H
#define PY4PD_H
#include <m_pd.h>
#include <g_canvas.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <pthread.h>

#ifdef _WIN64 // If windows 64bits include 
    #include <windows.h>
#endif
#ifdef HAVE_UNISTD_H // from Pure Data source code
    #include <unistd.h>
#endif

// Python include
#define PY_SSIZE_T_CLEAN // Good practice to use this
#include <Python.h>

// Pd Module for Python
// #include "module.h"


/* 


TODO: Way to set global variables, I think that will be important for things like general path (lilypond, etc)

TODO: Reset the function (like panic for sfont~).
    TODO: Add a way all functions to be reseted.

TODO: Add some way to run list how arguments
    FORMULA I: Work with [1 2 3 4 5] and transform this to a list for Python
    FORMULA II: MAKE SOME SPECIAL OBJECT TO WORK WITH LISTS 
    
TODO: If the run method set before the end of the thread, there is an error, that close all PureData.

TODO: The set function need to run after the load of the object, but before the start of the thread???
        It does not load external modules in the pd_new.

*/

// =====================================
typedef struct _py { // It seems that all the objects are some kind of class.
    t_object            x_obj; // convensao no puredata source code
    t_canvas            *x_canvas; // pointer to the canvas
 
    t_int                 state; // thread state
    t_int                 object_number; // object number
    t_inlet             *in1;

    // set global for variables
    PyObject            *globals;

    // define py_interpreter
    PyObject            *module; // python object
    PyObject            *function; // function name 
    PyInterpreterState  *py_main_interpreter;
    PyInterpreterState  *py_thread_interpreter;
    t_int               thread; // arguments
    t_int               function_called; // flag to check if the set function was called
    t_int               py_arg_numbers; // number of arguments
    t_symbol            *packages_path; // packages path 
    t_symbol            *home_path; // home path this always is the path folder (?)
    t_symbol            *object_path;
    t_symbol            *function_name; // function name
    t_symbol            *script_name; // script name
    
    t_outlet            *out_A; // outlet 1.
}t_py;


// PD GLOBAL OBJECT
// // ============================================

static t_py *py4pd_object;

// create a pointer for the t_py class

static t_class *py4pd_class;
static int object_count = 1;
static int thread_status[100];
static int running_some_thread = 0;
static int pyReinit = 0; // flag to check if the python interpreter was reinitialized
static PyInterpreterState *pymain = NULL; // main interpreter state

// create and array of pointers for the t_py class
static t_py *py4pd_object_array[100];


// Set Debug mode
#define DEBUG 1


#endif
