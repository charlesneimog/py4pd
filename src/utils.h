// clang-format off
#ifndef PY4PD_UTILS_H
#define PY4PD_UTILS_H

#include "py4pd.h"

int Py4pdUtils_ParseLibraryArguments(t_py *x, PyCodeObject *code, int *argc, t_atom **argv);
t_py *Py4pdUtils_GetObject(PyObject *pd_module);
void Py4pdUtils_ParseArguments(t_py *x, t_canvas *c, int argc, t_atom *argv);
char *Py4pdUtils_GetFolderName(char *path);
const char *Py4pdUtils_GetFilename(const char *path);
void Py4pdUtils_CheckPkgNameConflict(t_py *x, char *folderToCheck, t_symbol *script_file_name);
void Py4pdUtils_FindObjFolder(t_py *x);
void Py4pdUtils_CreateTempFolder(t_py *x);
void Py4pdUtils_GetEditorCommand(t_py *x, char *command, int line);
int Py4pdUtils_ExecuteSystemCommand(const char *command);
int Py4pdUtils_IsNumericOrDot(const char *str);
void Py4pdUtils_RemoveChar(char *str, char c);
char *Py4pdUtils_Mtok(char *input, char *delimiter);
void Py4pdUtils_FromSymbolSymbol(t_py *x, t_symbol *s, t_outlet *outlet);
int Py4pdUtils_RunPy(t_py *x, PyObject *pArgs, PyObject *pDict);
t_py4pd_pValue *Py4pdUtils_Run(t_py *x, PyObject *pArgs, t_py4pd_pValue *pValuePointer);
PyObject *Py4pdUtils_RunPyAudioOut(t_py *x, PyObject *pArgs, PyObject *pKwargs);
void Py4pdUtils_PrintError(t_py *x);
void *Py4pdUtils_ConvertToPd(t_py *x, t_py4pd_pValue *pValue, t_outlet *outlet);
PyObject *Py4pdUtils_ConvertToPy(PyObject *listsArrays[], int argc, t_atom *argv);
void Py4pdUtils_SetObjConfig(t_py *x);
void Py4pdUtils_AddPathsToPythonPath(t_py *x);
PyObject *Py4pdUtils_AddPdObject(t_py *x);
void Py4pdUtils_ReadGifFile(t_py *x, const char *filename);
void Py4pdUtils_ReadPngFile(t_py *x, const char *filename);
uint32_t Py4pdUtils_Ntohl(uint32_t netlong);
void *Py4pdUtils_FreeObj(t_py *x);
void Py4pdUtils_CreatePicObj(t_py *x, PyObject *PdDict, t_class *object_PY4PD_Class, int argc, t_atom *argv);
void Py4pdUtils_CopyPy4pdValueStruct(t_py4pd_pValue *src, t_py4pd_pValue *dest);

#endif

