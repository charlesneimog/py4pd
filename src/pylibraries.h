#ifndef PYLIBRARY_MODULE_H
#define PYLIBRARY_MODULE_H

#include "m_pd.h"
#include "py4pd.h"

//
// // =====================================
// typedef struct _pyObject { // It seems that all the objects are some kind of class.
//     t_object            x_obj; // convensao no puredata source code
//
//     PyObject            *module; // script name
//     PyObject            *function; // function name
//     PyObject            *Dict; // parameters
//
//     // Canvas
//     t_canvas            *x_canvas; // pointer to the canvas
//
//     // object pathnames
//     t_symbol            *home_path; // path to the python script
//     t_symbol            *py4pd_scripts; // path to the python script
//     t_symbol            *py4pd_folder; // path to the python script
//     t_symbol            *packages_path; // path to the python script
//
//     // verifications
//     t_int               function_called; 
//     t_int               py_arg_numbers; 
//     
//     // Create one inlet
//     t_inlet         *in1; // intlet 1
//     t_outlet        *out1; // outlet 1
//
//
// }t_pyObject;

// =====================================

extern void *py_newObject(t_symbol *s, int argc, t_atom *argv);
void *py_freeObject(t_py *x);


extern PyObject *pdAddPyObject(PyObject *self, PyObject *args);

#endif
