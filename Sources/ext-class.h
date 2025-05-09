// clang-format off
#ifndef PY4PD_CLASS_H
#define PY4PD_CLASS_H

#include "py4pd.h"

extern PyTypeObject Py4pdNewObj_Type;

typedef struct {
    PyObject_HEAD const char *objName;
    int objType;
    int objIsPlayable;
    PyObject *objFigSize;
    int objOutPyObjs;
    int noOutlets;
    int auxOutlets;
    int requireNofOutlets;
    int ignoreNoneOutputs;
    int allowEdit;
    const char *objImage;
    const char *helpPatch;

    // PyFunctions
    PyObject *pFuncFloat;
    PyObject *pFuncSymbol;
    PyObject *pFuncList;
    PyObject *pFuncAnything;
    PyObject *pFuncBang;

    // Audio
    PyObject *pAudio;

    // PySel Methods
    PyObject *pDictSelectors; // dict with method and function
    PyObject *pSelectorArgs; // dict with method args

    // PyType Method
    PyObject *pDictTypes; // dict with method and function
    PyObject *pTypeArgs; // dict with method args
    unsigned int pObjMethod;

} Py4pdNewObj;



#endif
