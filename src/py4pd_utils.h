#ifndef PY4PD_UTILITIES_H
#define PY4PD_UTILITIES_H

#include "py4pd.h"

#ifdef __linux__
    #define __USE_GNU
#endif

#include <dlfcn.h>


// declare function of utilities.c
void findpy4pd_folder(t_py *x);
void py4pd_tempfolder(t_py *x);
int *set_py4pd_config(t_py *x);
char *get_editor_command(t_py *x);
void pd4py_system_func(const char *command);
int isNumericOrDot(const char *str);
void removeChar(char *str, char c);
void *py4pd_convert_to_pd(t_py *x, PyObject *pValue);
// void *py4pd_convert_to_py(PyObject *listsArrays[], int argc, t_atom *argv);
PyObject *py4pd_convert_to_py(PyObject *listsArrays[], int argc, t_atom *argv);
int *set_py4pd_config(t_py *x);


#endif
