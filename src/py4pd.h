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

/* 
TODO: Way to set global variables, I think that will be important for things like general path (lilypond, etc)

TODO: Reset the function (like panic for sfont~), In some calls seems that the *function become NULL? 

TODO: make function home work with spaces, mainly for Windows OS where the use of lilypond in python need to be specified with spaces

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
    PyObject            *module; // python object
    PyObject            *function; // function name  
    int                 *state; // thread state
    int                 *object_number; // object number
    t_inlet             *in1;

    // set global for variables
    PyObject            *globals;

    // define py_interpreter
    PyInterpreterState   *py_interpreter;
    t_float             x_f;
    t_float             *thread; // arguments
    t_float             *function_called; // flag to check if the set function was called
    t_symbol            *packages_path; // packages path 
    t_symbol            *home_path; // home path this always is the path folder (?)
    t_symbol            *object_path;
    t_symbol            *function_name; // function name
    t_symbol            *script_name; // script name
    t_float             *py_arg_numbers; // number of arguments
    t_outlet            *out_A; // outlet 1.
}t_py;


// PD GLOBAL OBJECT
// // ============================================

static t_py *py4pd_object;

static PyInterpreterState *pymain = NULL; // main interpreter state

#endif
