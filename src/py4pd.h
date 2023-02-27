#ifndef PY4PD_H
#define PY4PD_H
#include <m_pd.h>
#include <pthread.h>
#ifdef _WIN64 
    #include <windows.h>  // on Windows, system() open a console window and we don't want that
#endif
#define PY_SSIZE_T_CLEAN // Good practice to use this before include Python.h because it will remove some deprecated function
#include <Python.h>

// Include numpy
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// =====================================
typedef struct _py { // It seems that all the objects are some kind of class.
    t_object            x_obj; // convensao no puredata source code
    t_float             py4pd_audio; // audio

    PyObject            *module; // python object
    PyObject            *function; // function name
    t_int               object_number; // object number
    t_int               thread; // arguments
    t_int               pictureMode; // picture mode
    t_int               function_called; // flag to check if the set function was called
    t_int               py_arg_numbers; // number of arguments
    t_int               use_NumpyArray; // flag to check if is to use numpy array in audioInput
    t_int               audioOutput; // flag to check if is to use audio output

    
    t_canvas            *x_canvas; // pointer to the canvas
    t_inlet             *in1; // intlet 1
    t_symbol            *editorName; // editor name
    t_symbol            *packages_path; // packages path 
    t_symbol            *home_path; // home path this always is the path folder (?)
    t_symbol            *object_path; // save object path   TODO: want to save scripts inside this folder and make all accessible
    t_symbol            *function_name; // function name
    t_symbol            *script_name; // script name
    t_outlet            *out_A; // outlet 1.
}t_py;

// Set Debug mode
#define DEBUG 1

#endif
