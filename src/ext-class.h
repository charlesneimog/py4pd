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
    const char *objImage;
    const char *helpPatch;

    // PyFunctions
    PyObject *pFuncFloat;
    PyObject *pFuncSymbol;
    PyObject *pFuncList;
    PyObject *pFuncAnything;
    PyObject *pFuncBang;
    PyObject *pAudioIn;
    PyObject *pAudioOut;
    PyObject *pAudio;

    // PySel Methods
    PyObject *pDictSelectors; // dict with method and function
    PyObject *pSelectorArgs; // dict with method args

} Py4pdNewObj;



#endif
