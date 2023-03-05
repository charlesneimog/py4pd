#ifndef PY4PD_UTILITIES_H
#define PY4PD_UTILITIES_H

#include "py4pd.h"

// declare function of utilities.c
int createHiddenFolder(t_py *x); 
int *set_py4pd_config(t_py *x);
char *get_editor_command(t_py *x);
int isNumericOrDot(const char *str);
char *removeChar(char *str, char garbage);
void *py4pd_convert_to_pd(t_py *x, PyObject *pValue);
void *py4pd_convert_to_py(PyObject *listsArrays[], int argc, t_atom *argv);
int *set_py4pd_config(t_py *x);
void *findpy4pd_folder(t_py *x, t_symbol *s, int argc, t_atom *argv);

#endif
