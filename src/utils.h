#ifndef PY4PD_UTILITIES_H
#define PY4PD_UTILITIES_H

#include "py4pd.h"

#ifdef __linux__
    #define __USE_GNU
#endif

#include <dlfcn.h>

// ========================
int parseLibraryArguments(t_py *x, PyCodeObject *code, int argc, t_atom *argv);
void parsePy4pdArguments(t_py *x, t_canvas *c, int argc, t_atom *argv);
void findPy4pdFolder(t_py *x);
void createPy4pdTempFolder(t_py *x);
void setPy4pdConfig(t_py *x);
char *getEditorCommand(t_py *x);
void executeSystemCommand(const char *command);
int isNumericOrDot(const char *str);
void removeChar(char *str, char c);

// ========================
t_py *get_py4pd_object(void);
// ========================
char* get_folder_name(char* path);
const char* get_filename(const char* path);

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
