#include "py4pd.h"

#include "utils.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PY4PD_NUMPYARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#include <numpy/arrayobject.h>

static void Py4pdAudio_Audio2PdAudio(t_py *x, PyObject *pValue,
                                     t_sample *audioOut, int numChannels,
                                     int n) {
    if (pValue != NULL) {
        if (PyArray_Check(pValue)) {

            PyArrayObject *pArray =
                PyArray_GETCONTIGUOUS((PyArrayObject *)pValue);
            PyArray_Descr *pArrayType =
                PyArray_DESCR(pArray); // double or float
            int arrayLength = PyArray_SIZE(pArray);
            int returnedChannels = PyArray_NDIM(pArray);

            if (x->nChs < returnedChannels) {
                pd_error(x,
                         "[%s] Returned %d channels, but the object has just "
                         "%d channels",
                         x->objName->s_name, returnedChannels, (int)x->nChs);
                x->audioError = 1;
                return;
            }

            if (arrayLength <= n) {
                if (pArrayType->type_num == NPY_FLOAT) {
                    float *pArrayData = (float *)PyArray_DATA(pArray);
                    int i;
                    for (i = 0; i < arrayLength; i++) {
                        audioOut[i] = pArrayData[i];
                    }

                } else if (pArrayType->type_num == NPY_DOUBLE) {
                    double *pArrayData = (double *)PyArray_DATA(pArray);
                    int i;
                    for (i = 0; i < arrayLength; i++) {
                        audioOut[i] = pArrayData[i];
                    }
                } else {
                    pd_error(x,
                             "[%s] The numpy array must be float or "
                             "double, returned %d",
                             x->objName->s_name, pArrayType->type_num);
                    x->audioError = 1;
                }
            } else {
                pd_error(x,
                         "[%s] The numpy array return more channels "
                         "that the object was created. It has %d channels "
                         "returned %d channels",
                         x->objName->s_name, numChannels,
                         (int)arrayLength / (int)x->vectorSize);
                x->audioError = 1;
            }
            Py_DECREF(pArray);
        } else {
            pd_error(x, "[%s] Python function must return a numpy array",
                     x->objName->s_name);
            x->audioError = 1;
        }
    } else {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[%s] Call failed: %s", x->objName->s_name,
                 PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        PyErr_Clear();
        for (int i = 0; i < numChannels; i++) {
            for (int j = 0; j < x->vectorSize; j++) {
                audioOut[i * x->vectorSize + j] = 0;
            }
        }
        x->audioError = 1;
    }
}

// =====================================
t_int *Py4pdAudio_AudioINPerform(t_int *w) {
    t_py *x = (t_py *)(w[1]);

    if (x->audioError) {
        return (w + 4);
    }

    t_sample *AudioIn = (t_sample *)(w[2]);
    int n = (int)(w[3]);
    int numChannels = n / x->vectorSize;
    npy_intp dims[] = {numChannels, x->vectorSize};
    PyObject *pAudio = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, AudioIn);

    // NOTE: pAudio must use NPY_DOUBLE in pd64
    if (x->pArgTuple == NULL) {
        x->pArgTuple = PyTuple_New(x->pArgsCount);
        return (w + 4);
    }
    PyTuple_SetItem(x->pArgTuple, 0, pAudio);
    for (int i = 1; i < x->pArgsCount; i++) {
        PyTuple_SetItem(x->pArgTuple, i, x->pyObjArgs[i]->pValue);
        Py_INCREF(x->pyObjArgs[i]->pValue);
    }

    Py4pdUtils_RunPy(x, x->pArgTuple, x->kwargsDict);
    return (w + 4);
}

// =====================================
t_int *Py4pdAudio_AudioOUTPerform(t_int *w) {
    t_py *x = (t_py *)(w[1]);

    if (x->audioError) {
        return (w + 4);
    }

    t_sample *audioOut = (t_sample *)(w[2]);
    int n = (int)(w[3]);
    PyObject *pValue;
    int numChannels = x->nChs;

    if (x->pArgTuple == NULL) {
        x->pArgTuple = PyTuple_New(x->pArgsCount);
        return (w + 4);
    }
    for (int i = 0; i < x->pArgsCount; i++) {
        PyTuple_SetItem(x->pArgTuple, i, x->pyObjArgs[i]->pValue);
        Py_INCREF(x->pyObjArgs[i]->pValue);
    }

    pValue = Py4pdUtils_RunPyAudioOut(x, x->pArgTuple, NULL);
    Py4pdAudio_Audio2PdAudio(x, pValue, audioOut, numChannels, n);
    Py_XDECREF(pValue);
    return (w + 4);
}

// =====================================
t_int *Py4pdAudio_AudioPerform(t_int *w) {
    t_py *x = (t_py *)(w[1]);

    if (x->audioError)
        return (w + 5);
    t_sample *audioIn = (t_sample *)(w[2]);
    t_sample *audioOut = (t_sample *)(w[3]);
    int n = (int)(w[4]);
    PyObject *pValue;
    int numChannels = x->nChs;
    if (x->pArgTuple == NULL) {
        x->pArgTuple = PyTuple_New(x->pArgsCount);
        return (w + 5);
    }
    npy_intp dims[] = {numChannels, x->vectorSize};
    PyObject *pAudio = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, audioIn);
    if (x->pArgTuple == NULL)
        return (w + 5);
    PyTuple_SetItem(x->pArgTuple, 0, pAudio);
    for (int i = 1; i < x->pArgsCount; i++) {
        PyTuple_SetItem(x->pArgTuple, i, x->pyObjArgs[i]->pValue);
        Py_INCREF(x->pyObjArgs[i]->pValue);
    }
    pValue = Py4pdUtils_RunPyAudioOut(x, x->pArgTuple, NULL);
    if (pValue == NULL) {
        x->audioError = 1;
        return (w + 5);
    }
    Py4pdAudio_Audio2PdAudio(x, pValue, audioOut, numChannels, n);

    Py_XDECREF(pValue);

    return (w + 5);
}

// =====================================
void Py4pdAudio_Dsp(t_py *x, t_signal **sp) {
    if (x->objType == PY4PD_AUDIOINOBJ) {
        x->nChs = sp[0]->s_nchans;
        x->vectorSize = sp[0]->s_n;
        dsp_add(Py4pdAudio_AudioINPerform, 3, x, sp[0]->s_vec,
                PY4PDSIGTOTAL(sp[0]));
    } else if (x->objType == PY4PD_AUDIOOUTOBJ) {
        x->vectorSize = sp[0]->s_n;
        signal_setmultiout(&sp[0], x->nChs);
        dsp_add(Py4pdAudio_AudioOUTPerform, 3, x, sp[0]->s_vec,
                PY4PDSIGTOTAL(sp[0]));
    } else if (x->objType == PY4PD_AUDIOOBJ) {
        x->nChs = sp[0]->s_nchans;
        x->vectorSize = sp[0]->s_n;
        signal_setmultiout(&sp[1], sp[0]->s_nchans);
        dsp_add(Py4pdAudio_AudioPerform, 4, x, sp[0]->s_vec, sp[1]->s_vec,
                PY4PDSIGTOTAL(sp[0]));
    }
}
