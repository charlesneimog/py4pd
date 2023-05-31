#ifndef PY4PD_UTILITIES_H
#define PY4PD_UTILITIES_H

#include "py4pd.h"

#ifdef __linux__
    #define __USE_GNU
#endif

#include <dlfcn.h>

// ========================
void create_pyObject_inlets(t_py *x, int argc, t_atom *argv);
int libraries_parser_args(t_py *x, PyCodeObject *code, int argc, t_atom *argv);
void py4pd_parser_args(t_py *x, t_canvas *c, int argc, t_atom *argv);
void findpy4pd_folder(t_py *x);
void py4pd_tempfolder(t_py *x);
void set_py4pd_config(t_py *x);
char *get_editor_command(t_py *x);
void pd4py_system_func(const char *command);
int isNumericOrDot(const char *str);
void removeChar(char *str, char c);

// ========================
void *py4pd_convert_to_pd(t_py *x, PyObject *pValue);
PyObject *py4pd_convert_to_py(PyObject *listsArrays[], int argc, t_atom *argv);
PyObject *py4pd_add_pd_object(t_py *x);
void *pyobject_to_pointer(PyObject *pValue);
PyObject *pointer_to_pyobject(void *p);
void free_pyobject_data(void *p);

// ========================
void py4pd_fromsymbol_symbol(t_py *x, t_symbol *s);
uint32_t py4pd_ntohl(uint32_t netlong);

#endif
