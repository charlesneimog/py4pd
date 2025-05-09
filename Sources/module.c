#include "ext-class.h"
#include "ext-libraries.h"
#include "m_pd.h"
#include "pic.h"
#include "player.h"
#include "py4pd.h"
#include "utils.h"

#define PY_ARRAY_UNIQUE_SYMBOL PY4PD_NUMPYARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#include <numpy/arrayobject.h>

// =================================
/**
 * @brief Dict to save data for object varibles.
 */
static pdcollectHash *Py4pdMod_CreatePdcollectHash(int size) {
    pdcollectHash *hash_table = (pdcollectHash *)malloc(sizeof(pdcollectHash));
    hash_table->Size = size;
    hash_table->Count = 0;
    hash_table->Itens = (pdcollectItem **)calloc(size, sizeof(pdcollectItem *));
    return hash_table;
}

// =================================
/**
 * @brief Hash function to get the index of the hash table.
 */
static unsigned int Py4pdMod_HashFunction(pdcollectHash *hash_table, char *key) {
    // FAKE HASH FUNCTION
    int keyAlreadyExists = 0;
    for (int i = 0; i < hash_table->Size; i++) {
        if (hash_table->Itens[i] != NULL) {
            if (strcmp(hash_table->Itens[i]->Key, key) == 0) {
                return i;
            }
        }
    }
    if (keyAlreadyExists == 0) {
        for (int i = 0; i < hash_table->Size; i++) {
            if (hash_table->Itens[i] == NULL) {
                return i;
            }
        }
    }
    PyErr_SetString(PyExc_MemoryError, "[Python] pd.setglobalvar: memory error");
    return 0;
}

// =================================
/**
 * @brief Insert item in the hash table.
 */
static void Py4pdMod_InsertItem(pdcollectHash *hash_table, char *key, PyObject *source) {
    unsigned int index = Py4pdMod_HashFunction(hash_table, key);
    pdcollectItem *item = hash_table->Itens[index];

    // deep copy of source
    PyObject *obj = Py4pdUtils_DeepCopy(source);

    if (item == NULL && hash_table->Count <= hash_table->Size) {
        item = (pdcollectItem *)malloc(sizeof(pdcollectItem));
        item->Key = strdup(key);
        item->pItem = obj;
        hash_table->Itens[index] = item;
        item->aCumulative = 0;
        hash_table->Count++;
        return;
    } else if (hash_table->Count > hash_table->Size) {
        // WARNING: Untested code
        pd_error(0, "[Python] pd.setglobalvar: memory error, please report, code 001");
        // pdcollectItem **items = hash_table->items;
        // int size = hash_table->size;
        // hash_table = Py4pdMod_CreatePdcollectHash(size * 2);
        // for (int i = 0; i < size; i++) {
        //     if (items[i] != NULL) {
        //         Py4pdMod_InsertItem(hash_table, items[i]->key, items[i]->pItem);
        //     }
        // }
        // Py4pdMod_InsertItem(hash_table, key, obj);
        // for (int i = 0; i < size; i++) {
        //     if (items[i] != NULL) {
        //         free(items[i]->key);
        //         free(items[i]);
        //     }
        // }
        return;
    } else if (item != NULL) {
        // Py_INCREF(obj);
        Py_XDECREF(item->pItem);
        item->pItem = obj;
        return;
    }
}

// =================================
/**
 * @brief Insert item in the hash table, add to the end of a list.
 */
static void Py4pdMod_AccumItem(pdcollectHash *hash_table, char *key, PyObject *source) {
    unsigned int index = Py4pdMod_HashFunction(hash_table, key);
    pdcollectItem *item = hash_table->Itens[index];
    // PyObject *obj = Py4pdUtils_DeepCopy(source);
    if (item == NULL && hash_table->Count <= hash_table->Size) {
        item = (pdcollectItem *)malloc(sizeof(pdcollectItem));
        item->Key = strdup(key);
        item->pList = PyList_New(0);
        hash_table->Itens[index] = item;
    } else if (hash_table->Count > hash_table->Size) {
        PyErr_SetString(PyExc_MemoryError, "[Python] pd.setglobalvar: memory error");
        return;
    }
    if (item->WasCleaned) {
        item->WasCleaned = 0;
    }
    PyList_Append(item->pList, source);
    return;
}

// =================================
/**
 * @brief Clear/free item in the hash table.
 */
static void Py4pdMod_ClearItem(pdcollectHash *hash_table, char *key) {
    unsigned int index = Py4pdMod_HashFunction(hash_table, key);
    pdcollectItem *item = hash_table->Itens[index];
    if (item == NULL) {
        return;
    }
    if (item->WasCleaned)
        return;
    item->WasCleaned = 1;
    free(item->Key);
    Py_DECREF(item->pItem);
    free(item);
    hash_table->Itens[index] = NULL;
}

// =================================
/**
 * @brief Clear/free item in the hash table.
 */
static void Py4pdMod_ClearList(pdcollectHash *hash_table, char *key) {

    unsigned int index = Py4pdMod_HashFunction(hash_table, key);
    pdcollectItem *item = hash_table->Itens[index];
    if (item == NULL) {
        return;
    }
    if (item->WasCleaned) {
        return;
    }
    item->WasCleaned = 1;
    free(item->Key);
    // Py_DECREF(item->pList);
    free(item);
    hash_table->Itens[index] = NULL;
}

// =================================
/**
 * @brief Get item in the hash table.
 */
static pdcollectItem *Py4pdMod_GetObjArr(pdcollectHash *hash_table, char *key) {
    unsigned int index = Py4pdMod_HashFunction(hash_table, key);
    pdcollectItem *item = hash_table->Itens[index];
    if (item == NULL)
        return NULL;
    return item;
}

// =================================
/**
 * @brief Free item in the hash table.
 */
static void Py4pdMod_FreePdcollectItem(pdcollectItem *item) {
    if (item == NULL) {
        return;
    }
    if (item->WasCleaned) {
        return;
    }
    item->WasCleaned = 1;
    free(item->Key);

    if (item->pList) {
        Py_DECREF(item->pList);
    } else if (item->pItem) {
        Py_DECREF(item->pItem);
    }
    free(item);
}

// =================================
/**
 * @brief Free the hash table
 */
void Py4pdMod_FreePdcollectHash(pdcollectHash *hash_table) {
    if (hash_table == NULL) {
        return;
    }
    for (int i = 0; i < hash_table->Size; ++i) {
        Py4pdMod_FreePdcollectItem(hash_table->Itens[i]);
    }
    free(hash_table->Itens);
    free(hash_table);
}

// =================================
// ========== pd module ============
// =================================
/**
 * @brief Funcion to set a obj variable
 */
static PyObject *Py4pdMod_SetObjVar(PyObject *self, PyObject *args) {
    (void)self;
    t_py *x = Py4pdUtils_GetObject(self);
    if (x == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "[Python] pd.setglobalvar: py4pd is NULL");
        return NULL;
    }
    char *key;
    char *varName;
    PyObject *pValueScript;
    if (!PyArg_ParseTuple(args, "sO", &varName, &pValueScript)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.setglobalvar: wrong arguments");
        return NULL;
    }

    key = malloc(strlen(varName) + 40);
    pd_snprintf(key, strlen(varName) + 40, "%s_%p", varName, x);
    if (x->PdCollect == NULL) {
        x->PdCollect = Py4pdMod_CreatePdcollectHash(8);
    }
    Py4pdMod_InsertItem(x->PdCollect, key, pValueScript);
    free(key);
    Py_RETURN_TRUE;
}

// =================================
/**
 * @brief Funcion to get a obj variable
 */
static PyObject *Py4pdMod_GetObjVar(PyObject *self, PyObject *args, PyObject *keywords) {
    (void)self;

    t_py *x = Py4pdUtils_GetObject(self);
    if (x == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "[Python] pd.setglobalvar: py4pd is NULL");
        return NULL;
    }
    char *key;
    char *varName;

    if (!PyArg_ParseTuple(args, "s", &varName)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.setglobalvar: wrong arguments");
        return NULL;
    }

    key = malloc(strlen(varName) + 40);
    pd_snprintf(key, strlen(varName) + 40, "%s_%p", varName, x);
    if (x->PdCollect == NULL) {
        x->PdCollect = Py4pdMod_CreatePdcollectHash(8);
    }

    pdcollectItem *item = Py4pdMod_GetObjArr(x->PdCollect, key);

    if (keywords != NULL) {
        PyObject *pString = PyUnicode_FromString("initial_value");
        PyObject *pValueInit = PyDict_GetItem(keywords, pString);
        Py_DECREF(pString);
        if (pValueInit != NULL && item == NULL) {
            Py4pdMod_InsertItem(x->PdCollect, key, pValueInit);
            free(key);
            Py_INCREF(pValueInit);
            return pValueInit;
        }
    }

    free(key);
    if (item == NULL) {
        Py_RETURN_NONE;
    }

    if (item->aCumulative) {
        // Py_INCREF(item->pList);
        return item->pList;
    } else {
        Py_INCREF(item->pItem);
        return item->pItem;
    }
}

// =================================
/**
 * @brief Funcion to accumulate a obj variable
 */
static PyObject *Py4pdMod_AccumObjVar(PyObject *self, PyObject *args) {

    (void)self;
    t_py *x = Py4pdUtils_GetObject(self);
    if (x == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "[Python] pd.setglobalvar: py4pd is NULL");
        return NULL;
    }
    char *varName;
    PyObject *pValueScript;
    if (!PyArg_ParseTuple(args, "sO", &varName, &pValueScript)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.setglobalvar: wrong arguments");
        return NULL;
    }
    char *key = malloc(strlen(varName) + 40);
    pd_snprintf(key, strlen(varName) + 40, "%s_%p", varName, x);
    if (x->PdCollect == NULL) {
        x->PdCollect = Py4pdMod_CreatePdcollectHash(8);
    }
    Py4pdMod_AccumItem(x->PdCollect, key, pValueScript);
    x->PdCollect->Itens[Py4pdMod_HashFunction(x->PdCollect, key)]->aCumulative = 1;
    free(key);
    Py_RETURN_TRUE;
}

// =================================
/**
 * @brief Funcion to clear a obj variable
 */
static PyObject *Py4pdMod_ClearObjVar(PyObject *self, PyObject *args) {
    (void)self;

    t_py *x = Py4pdUtils_GetObject(self);
    if (x == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "[Python] pd.setglobalvar: py4pd is NULL");
        return NULL;
    }
    char *key;
    char *varName;

    if (!PyArg_ParseTuple(args, "s", &varName)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.setglobalvar: wrong arguments");
        return NULL;
    }

    key = malloc(strlen(varName) + 40);
    pd_snprintf(key, strlen(varName) + 40, "%s_%p", varName, x);
    if (x->PdCollect == NULL) {
        free(key);
        Py_RETURN_TRUE;
    }
    pdcollectItem *objArr = Py4pdMod_GetObjArr(x->PdCollect, key);
    if (objArr != NULL) {
        if (objArr->WasCleaned) {
            free(key);
            Py_RETURN_TRUE;
        }
    } else {
        free(key);
        Py_RETURN_TRUE;
    }

    if (objArr->aCumulative)
        Py4pdMod_ClearList(x->PdCollect, key);
    else
        Py4pdMod_ClearItem(x->PdCollect, key);

    free(key);
    Py_RETURN_TRUE;
}

// ======================================
/**
 * @brief Function that get the args for the obj (for example [py4pd -lib py4pd]
 * will return ['-lib', 'py4pd'])
 */
static PyObject *Py4pdMod_GetObjArgs(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    t_py *x = Py4pdUtils_GetObject(self);
    if (x == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "[Python] pd.setglobalvar: py4pd is NULL");
        return NULL;
    }

    // BUG:, when using -a this give the wrong result;
    PyObject *pList = PyList_New(0);
    for (int i = 0; i < x->ObjArgsCount; i++) {
        if (x->PdObjArgs[i].a_type == A_SYMBOL) {
            t_symbol *key = atom_getsymbolarg(i, x->ObjArgsCount, x->PdObjArgs);
            PyObject *strObj = PyUnicode_FromString(key->s_name);
            PyList_Append(pList, strObj);
            Py_DECREF(strObj);
        } else if (x->PdObjArgs[i].a_type == A_FLOAT) {
            int isInt = atom_getintarg(i, x->ObjArgsCount, x->PdObjArgs) ==
                        atom_getfloatarg(i, x->ObjArgsCount, x->PdObjArgs);
            t_float key = atom_getfloatarg(i, x->ObjArgsCount, x->PdObjArgs);
            if (isInt) {
                PyObject *Number = PyLong_FromLong(key);
                PyList_Append(pList, Number);
                Py_DECREF(Number);
            } else {
                PyObject *Number = PyFloat_FromDouble(key);
                PyList_Append(pList, Number);
                Py_DECREF(Number);
            }
        }
    }
    return pList;
}

// ======================================
/**
 * @brief This allows bypass the recursive calls in PureData
 */
static void Py4pdMod_RecursiveTick(t_py *x) {
    t_py4pd_pValue *pdPyValue = (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
    pdPyValue->pValue = x->RecursiveObject;
    Py4pdUtils_ConvertToPd(x, pdPyValue, x->MainOut);
    Py_DECREF(pdPyValue->pValue); // delete thing
    free(pdPyValue);
    Py_LeaveRecursiveCall();
}

// ======================================
/**
 * @brief This is the Python part to bypass the recursive calls in PureData
 */
static PyObject *Py4pdMod_PdRecursiveCall(PyObject *self, PyObject *args) {

    (void)self;

    PyObject *pValue;
    if (!PyArg_ParseTuple(args, "O", &pValue)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.setglobalvar: wrong arguments");
        return NULL;
    }

    t_py *x = Py4pdUtils_GetObject(self);
    x->RecursiveCalls++;

    Py_EnterRecursiveCall("[py4pd] Exceeded maximum recursion depth");
    if (x->RecursiveCalls == x->StackLimit) { // seems to be the limit in PureData,
        x->RecursiveObject = pValue;
        if (x->RecursiveClock == NULL)
            x->RecursiveClock = clock_new(x, (t_method)Py4pdMod_RecursiveTick);
        Py_INCREF(pValue); // avoid thing to be deleted
        x->RecursiveCalls = 0;
        clock_delay(x->RecursiveClock, 0);
        Py_RETURN_TRUE;
    }

    if (x == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "[pd._recursive] pd.recursive: py4pd is NULL");
        return NULL;
    }

    t_py4pd_pValue *pdPyValue = (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
    pdPyValue->pValue = pValue;
    Py4pdUtils_ConvertToPd(x, pdPyValue, x->MainOut);
    free(pdPyValue);
    Py_LeaveRecursiveCall();
    x->RecursiveCalls = 0;
    Py_RETURN_TRUE;
}

// ======================================
/**
 * @brief Get the number of outlets of a object
 */
static PyObject *Py4pdMod_PdGetOutletsCount(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    t_py *x = Py4pdUtils_GetObject(self);
    if (x == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "[Python] py4pd capsule not found. The module pd must "
                                            "be used inside py4pd object or functions.");
        return NULL;
    }
    if (x->ExtrasOuts == NULL) {
        return PyLong_FromLong(1);
    }
    return PyLong_FromLong(x->ExtrasOuts->outCount + 1);
}

// =================================
/**
 * @brief allows output data before return in Python Code.
 */
static PyObject *Py4pdMod_PdOut(PyObject *self, PyObject *args, PyObject *keywords) {

    (void)self;

    t_py *x = Py4pdUtils_GetObject(self);
    if (x == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "[Python] py4pd capsule not found. The module pd must "
                                            "be used inside py4pd object or functions.");
        return NULL;
    }

    PyObject *pValue;

    if (!PyArg_ParseTuple(args, "O", &pValue)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.out: wrong arguments");
        return NULL;
    }

    t_py4pd_pValue *pdPyValue = (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
    pdPyValue->pValue = pValue;
    pdPyValue->PdOutCount = 1;

    if (keywords != NULL && x->ExtrasOuts != NULL) {
        PyObject *outletNumber =
            PyDict_GetItemString(keywords, "out_n"); // it gets the data type output
        if (outletNumber == NULL) {
            Py4pdUtils_ConvertToPd(x, pdPyValue, x->MainOut);
            free(pdPyValue);
            Py_RETURN_TRUE;
        }

        if (!PyLong_Check(outletNumber)) {
            PyErr_SetString(PyExc_TypeError, "[Python] pd.out: out_n must be an integer.");
            Py_DECREF(pValue);
            free(pdPyValue);
            return NULL;
        }
        int outletNumberInt = PyLong_AsLong(outletNumber);
        if (outletNumberInt == 0) {
            Py4pdUtils_ConvertToPd(x, pdPyValue, x->MainOut);
            free(pdPyValue);
            Py_RETURN_TRUE;
        } else {
            outletNumberInt--;
            if ((x->ExtrasOuts->outCount > 0) && (outletNumberInt < x->ExtrasOuts->outCount)) {
                Py4pdUtils_ConvertToPd(x, pdPyValue, x->ExtrasOuts[outletNumberInt].u_outlet);
                free(pdPyValue);
                Py_RETURN_TRUE;
            } else {
                PyErr_SetString(PyExc_TypeError,
                                "[Python] pd.out: Please check the number of outlets.");
                Py_DECREF(pValue);
                free(pdPyValue);
                return NULL;
            }
        }
    } else {
        Py4pdUtils_ConvertToPd(x, pdPyValue, x->MainOut);
        free(pdPyValue);
    }
    Py_RETURN_TRUE;
}
// =================================
/**
 * @brief Prints messages to pd console
 */
static PyObject *Py4pdMod_PdPrint(PyObject *self, PyObject *args, PyObject *keywords) {
    (void)self;
    int printPrefix = 1;
    int objPrefix = 1;

    t_py *x = Py4pdUtils_GetObject(self);
    if (x == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "[Python] py4pd capsule not found. The module pd must "
                                            "be used inside py4pd object or functions.");
        return NULL;
    }

    if (keywords == NULL) {
        printPrefix = 1;
    } else {
        printPrefix = PyDict_Contains(keywords, PyUnicode_FromString("show_prefix"));
        objPrefix = PyDict_Contains(keywords, PyUnicode_FromString("obj_prefix"));
        if (printPrefix == -1)
            pd_error(NULL, "[Python] pd.print: error in show_prefix argument.");
        else if (printPrefix == 1) {
            PyObject *resize_value = PyDict_GetItemString(keywords, "show_prefix");
            if (resize_value == Py_True)
                printPrefix = 1;
            else if (resize_value == Py_False)
                printPrefix = 0;
            else {
                pd_error(NULL, "[Python] pd.print: show_prefix argument must "
                               "be True or False.");
                printPrefix = 1;
            }
        } else
            printPrefix = 1;
        if (objPrefix == -1)
            post("[Python] pd.print: error in obj_prefix argument.");
        else if (objPrefix == 1) {
            post("[Python] pd.print: obj_prefix argument must be True "
                 "or "
                 "False.");
        } else {
            objPrefix = 0;
        }
    }

    if (PyTuple_Size(args) > 0) {
        if (printPrefix == 1) { //
            if (x->ObjName == NULL) {
                startpost("[Python]: ");
            } else {
                startpost("[%s]: ", x->ObjName->s_name);
            }
            sys_pollgui();
        }
        for (int i = 0; i < PyTuple_Size(args); i++) {
            PyObject *arg = PyTuple_GetItem(args, i);
            PyObject *str = PyObject_Str(arg);
            startpost(PyUnicode_AsUTF8(str));
            startpost(" ");
        }
        startpost("\n");
    }

    Py_RETURN_TRUE;
}

// =================================
/**
 * @brief Prints logs to pd console
 */
static PyObject *Py4pdMod_PdLogPost(PyObject *self, PyObject *args) {
    (void)self;
    int postlevel;
    char *string;

    if (PyArg_ParseTuple(args, "is", &postlevel, &string)) {
        logpost(NULL, postlevel, "%s", string);
        return PyLong_FromLong(0);
    } else {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.logpost works with strings and numbers.");
        return NULL;
    }
}

// =================================
/**
 * @brief Prints errors to pd console
 */
static PyObject *Py4pdMod_PdError(PyObject *self, PyObject *args) {
    (void)self;

    char *string;

    t_py *x = Py4pdUtils_GetObject(self);

    if (x->ObjType > PY4PD_VISOBJ) { // if audio object
        x->AudioError = 1;
    }

    if (PyArg_ParseTuple(args, "s", &string)) {
        if (x == NULL) {
            pd_error(NULL, "%s", string);
            PyErr_Clear();
            return PyLong_FromLong(0);
        }

        if (x->PyObject == 1) {
            pd_error(x, "[%s]: %s", x->ObjName->s_name, string);
        } else {
            if (x->pFuncName == NULL) {
                pd_error(x, "%s", string);
            } else {
                pd_error(x, "[%s]: %s", x->pFuncName->s_name, string);
            }
        }
        PyErr_Clear();
    } else {
        PyErr_SetString(PyExc_TypeError, "[Python] argument of pd.error must be a string");
        return NULL;
    }
    return PyLong_FromLong(0);
}

// =================================
/**
 * @brief Same that use the object [send] or [s] in pd
 */
static PyObject *Py4pdMod_PdSend(PyObject *self, PyObject *args) {
    (void)self;
    char *receiver;
    char *string;
    float floatNumber;
    int intNumber;
    PyObject *listargs;

    if (PyArg_ParseTuple(args, "ss", &receiver, &string)) {
        t_symbol *symbol = gensym(receiver);
        if (symbol->s_thing) {
            pd_symbol(symbol->s_thing, gensym(string));
        } else {
            post("[Python] pd.send not found object [r %s] in pd patch", receiver);
        }
    } else if (PyArg_ParseTuple(args, "sf", &receiver, &floatNumber)) {
        t_symbol *symbol = gensym(receiver);
        if (symbol->s_thing) {
            pd_float(symbol->s_thing, floatNumber);
        } else {
            post("[Python] pd.send not found object [r %s] in pd patch", receiver);
        }
    } else if (PyArg_ParseTuple(args, "si", &receiver, &intNumber)) {
        t_symbol *symbol = gensym(receiver);
        if (symbol->s_thing) {
            pd_float(symbol->s_thing, intNumber);
        } else {
            post("[Python] pd.send not found object [r %s] in pd patch", receiver);
        }
    } else if (PyArg_ParseTuple(args, "sO", &receiver, &listargs)) {
        if (PyDict_Check(listargs)) {
            char error_message[] = "[Python] pd.send received a type "
                                   "'dict', it must be a list, "
                                   "string, int, or float.";
            PyErr_SetString(PyExc_TypeError, error_message);
            return NULL;
        }
        t_atom *list_array;
        int list_size = PyList_Size(listargs);
        list_array = (t_atom *)malloc(list_size * sizeof(t_atom));
        int i;
        for (i = 0; i < list_size; ++i) {
            PyObject *pValue_i = PyList_GetItem(listargs, i);
            if (PyLong_Check(pValue_i)) {
                long result = PyLong_AsLong(pValue_i);
                float result_float = (float)result;
                SETFLOAT(&list_array[i], result_float);
            } else if (PyFloat_Check(pValue_i)) {
                double result = PyFloat_AsDouble(pValue_i);
                float result_float = (float)result;
                SETFLOAT(&list_array[i], result_float);
            } else if (PyUnicode_Check(pValue_i)) {
                const char *result = PyUnicode_AsUTF8(pValue_i);
                SETSYMBOL(&list_array[i], gensym(result));
            } else if (Py_IsNone(pValue_i)) {
                // Not possible represent None in PureData
            } else {
                char error_message[] = "[Python] received a type '%s' in index %d of the "
                                       "list, it must be a string, int, or float.";
                PyErr_SetString(PyExc_TypeError, error_message);
                Py_DECREF(pValue_i);
                Py_DECREF(args);
                free(list_array);
                return NULL;
            }
        }
        if (gensym(receiver)->s_thing) {
            pd_list(gensym(receiver)->s_thing, &s_list, list_size, list_array);
            free(list_array);
        } else {
            pd_error(NULL, "[Python] object [r %s] not found", receiver);
            free(list_array);
        }
    } else {
        char error_message[100];
        PyObject *pValue_i = PyTuple_GetItem(args, 1);
        pd_snprintf(error_message, MAXPDSTRING,
                    "[Python] pd.send received a type '%s', it must be a "
                    "string, int, or float.",
                    pValue_i->ob_type->tp_name);
        PyErr_SetString(PyExc_TypeError, error_message);
        return NULL;
    }
    PyErr_Clear();
    Py_RETURN_TRUE;
}

// =================================
/**
 * @brief Same that use the [tabwrite] in pd
 */
static PyObject *Py4pdMod_PdTabWrite(PyObject *self, PyObject *args, PyObject *keywords) {
    (void)self;
    int resize = 0;
    int vecsize;
    t_garray *pdarray;
    t_word *vec;
    char *string;
    PyObject *samples;

    if (keywords == NULL) {
        resize = 0;
        PyErr_Clear();
    } else {

        resize = PyDict_Contains(keywords, PyUnicode_FromString("resize"));
        if (resize == -1) {
            post("error");
        } else if (resize == 1) {
            PyObject *resize_value = PyDict_GetItemString(keywords, "resize");
            if (resize_value == Py_True) {
                resize = 1;
            } else if (resize_value == Py_False) {
                resize = 0;
            } else {
                resize = 0;
            }
        } else {
            resize = 0;
        }
    }

    // ================================
    t_py *x = Py4pdUtils_GetObject(self);
    if (x == NULL) {
        post("[Python] py4pd capsule not found. The module pd must be used "
             "inside py4pd object or functions.");
        return NULL;
    }

    if (PyArg_ParseTuple(args, "sO", &string, &samples)) {
        t_symbol *pd_symbol = gensym(string);
        if (!(pdarray = (t_garray *)pd_findbyclass(pd_symbol, garray_class))) {
            pd_error(x, "[Python] Array %s not found.", string);
            return NULL;
        } else if (!garray_getfloatwords(pdarray, &vecsize, &vec)) {
            pd_error(x, "[Python] Bad template for tabwrite '%s'.", string);
            return NULL;
        } else {
            int i;
            _import_array();
            if (samples == NULL) {
                pd_error(x, "[Python] pd.tabwrite: wrong arguments");
                return NULL;
            }
            if (PyList_Check(samples)) {
                if (resize == 1) {
                    garray_resize_long(pdarray, PyList_Size(samples));
                    vecsize = PyList_Size(samples);
                    garray_getfloatwords(pdarray, &vecsize, &vec);
                }
                for (i = 0; i < vecsize; i++) {
                    float result_float = (float)PyFloat_AsDouble(PyList_GetItem(samples, i));
                    vec[i].w_float = result_float;
                }
            } else if (PyArray_Check(samples)) {
                PyArrayObject *pArray = PyArray_GETCONTIGUOUS((PyArrayObject *)samples);
                PyArray_Descr *pArrayType = PyArray_DESCR(pArray); // double or float
                vecsize = PyArray_SIZE(pArray);
                if (resize == 1) {
                    garray_resize_long(pdarray, vecsize);
                    garray_getfloatwords(pdarray, &vecsize, &vec);
                }
                if (pArrayType->type_num == NPY_FLOAT) {
                    float *pArrayData = (float *)PyArray_DATA(pArray);
                    for (i = 0; i < vecsize; i++) {
                        vec[i].w_float = pArrayData[i];
                    }

                } else if (pArrayType->type_num == NPY_DOUBLE) {
                    double *pArrayData = (double *)PyArray_DATA(pArray);
                    for (i = 0; i < vecsize; i++) {
                        vec[i].w_float = pArrayData[i];
                    }
                } else {
                    pd_error(NULL,
                             "[%s] The numpy array must be float or "
                             "double, returned %d",
                             x->ObjName->s_name, pArrayType->type_num);
                    x->AudioError = 1;
                }

            }

            else {
                pd_error(x, "[Python] pd.tabwrite: wrong arguments");
                return NULL;
            }
            garray_redraw(pdarray);
            PyErr_Clear();
        }
    }
    Py_RETURN_TRUE;
}
// =================================
/**
 * @brief Same that use the [tabread] in pd
 */
static PyObject *Py4pdMod_PdTabRead(PyObject *self, PyObject *args, PyObject *keywords) {
    (void)self;
    int vecsize;
    t_garray *pdarray;
    t_word *vec;
    char *string;
    int numpy;

    // ================================
    t_py *x = Py4pdUtils_GetObject(self);
    // ================================

    if (keywords == NULL) {
        numpy = 1;
        int numpyArrayImported = _import_array();
        if (numpyArrayImported == 0) {
            x->NumpyImported = 1;
        } else {
            x->NumpyImported = 0;
            pd_error(x, "[py4pd] Not possible to import numpy array");
            return NULL;
        }

        PyErr_Clear();
    } else {
        numpy = PyDict_Contains(keywords, PyUnicode_FromString("numpy"));
        if (numpy == -1) {
            pd_error(NULL, "[Python] Check the keyword arguments.");
            return NULL;
        } else if (numpy == 1) {
            PyObject *numpy_value = PyDict_GetItemString(keywords, "numpy");
            if (numpy_value == Py_True) {
                numpy = 1;
            } else if (numpy_value == Py_False) {
                numpy = 0;
            } else {
                numpy = 0;
            }
        } else {
            numpy = 0;
        }
    }

    if (PyArg_ParseTuple(args, "s", &string)) {
        t_symbol *pd_symbol = gensym(string);
        if (!(pdarray = (t_garray *)pd_findbyclass(pd_symbol, garray_class))) {
            pd_error(x, "[Python] Array %s not found.", string);
            PyErr_SetString(PyExc_TypeError, "[Python] pd.tabread: array not found");
        } else {
            int i;
            if (numpy == 0) {
                garray_getfloatwords(pdarray, &vecsize, &vec);
                PyObject *pAudio = PyList_New(vecsize);
                for (i = 0; i < vecsize; i++) {
                    PyList_SetItem(pAudio, i, PyFloat_FromDouble(vec[i].w_float));
                }
                PyErr_Clear();
                return pAudio;
            } else if (numpy == 1) {
                garray_getfloatwords(pdarray, &vecsize, &vec);
                npy_intp dims[1] = {vecsize};
                PyObject *pAudio = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
                double *pArrayData = (double *)PyArray_DATA((PyArrayObject *)pAudio);
                if (pArrayData == NULL) {
                    pd_error(x, "[Python] pd.tabread: error creating "
                                "numpy array for vec");
                    return NULL;
                }
                for (i = 0; i < vecsize; i++) {
                    pArrayData[i] = vec[i].w_float;
                }
                return pAudio;

            } else {
                pd_error(x, "[Python] Check the keyword arguments.");
                return NULL;
            }
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.tabread: wrong arguments");
        return NULL;
    }
    return NULL;
}

// =================================
//
static void Py4pdMod_ScheduledOpenPatch(t_py *x) {

    t_atom *listArray;
    int listSize = 2;
    listArray = (t_atom *)malloc(listSize * sizeof(t_atom));
    const char *patchName = PyUnicode_AsUTF8(PyList_GetItem(x->pObjVarDict, 0));
    const char *patchDir = PyUnicode_AsUTF8(PyList_GetItem(x->pObjVarDict, 1));

    SETSYMBOL(&listArray[0], gensym(patchName));
    SETSYMBOL(&listArray[1], gensym(patchDir));

    if (gensym("pd")->s_thing) {
        typedmess(gensym("pd")->s_thing, gensym("open"), 2, listArray);
        clock_free(x->RecursiveClock);
        Py_DECREF(x->pObjVarDict);
        x->pObjVarDict = NULL;
        free(listArray);
        return;
    } else {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.open_patch: pd object not found");
        clock_free(x->RecursiveClock);
        Py_DECREF(x->pObjVarDict);
        x->pObjVarDict = NULL;
        free(listArray);
        return;
    }
}

// =================================
static PyObject *Py4pdMod_PipInstallRequirements(PyObject *self, PyObject *args) {
    t_py *x = Py4pdUtils_GetObject(self);
    char *requirements;

    if (!PyArg_ParseTuple(args, "s", &requirements)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd._pipinstall: wrong arguments.");
    }

    // check

    t_atom pdArgs[3];
    SETSYMBOL(&pdArgs[0], gensym("install"));
    SETSYMBOL(&pdArgs[1], gensym("-r"));
    SETSYMBOL(&pdArgs[2], gensym(requirements));
    x->PipGlobalInstall = 1;
    Py4pd_Pip(x, gensym("pip"), 3, pdArgs);
    Py_RETURN_TRUE;
}

// =================================
/**
 * @brief Same that use the [home] in pd
 */
static PyObject *Py4pdMod_OpenPatch(PyObject *self, PyObject *args) {
    (void)self;
    t_py *x = Py4pdUtils_GetObject(self);

    char *patchName;
    char *patchDir;

    if (!PyArg_ParseTuple(args, "ss", &patchName, &patchDir)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.open_patch: wrong arguments. Use "
                                         "PATCH_NAME followed by PATCH_DIR");
        return NULL;
    }

    PyObject *pLibraryPath = PyUnicode_FromString(patchDir);
    PyObject *sys_path = PySys_GetObject("path");
    PyList_Insert(sys_path, 0, pLibraryPath);
    Py_DECREF(pLibraryPath);

    PyObject *pList = PyList_New(2);
    PyList_SetItem(pList, 0, PyUnicode_FromString(patchName));
    PyList_SetItem(pList, 1, PyUnicode_FromString(patchDir));
    x->RecursiveClock = clock_new(x, (t_method)Py4pdMod_ScheduledOpenPatch);
    x->pObjVarDict = pList;
    clock_delay(x->RecursiveClock, 0);
    Py_RETURN_TRUE;
}

// =================================
/**
 * @brief Same that use the [home] in pd
 */
static PyObject *Py4pdMod_GetPatchHome(PyObject *self, PyObject *args) {
    (void)self;

    t_py *x = Py4pdUtils_GetObject(self);
    if (x == NULL) {
        post("[Python] py4pd capsule not found. The module pd must be used "
             "inside py4pd object or functions.");
        return NULL;
    }
    if (!PyArg_ParseTuple(args, "")) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.home: no argument expected");
        return NULL;
    }
    return PyUnicode_FromString(x->PdPatchPath->s_name);
}

// =================================
/**
 * @brief Returns where py4pd is installed
 */
static PyObject *Py4pdMod_GetPyObjectType(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *pValue;

    if (!PyArg_ParseTuple(args, "O", &pValue)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.getobjecttype: wrong arguments");
        return NULL;
    }
    const char *type = Py_TYPE(pValue)->tp_name;
    return PyUnicode_FromString(type);
}

// =================================
/**
 * @brief Returns where py4pd is installed
 */
static PyObject *Py4pdMod_GetPy4pdFolder(PyObject *self, PyObject *args) {
    (void)self;

    t_py *x = Py4pdUtils_GetObject(self);
    if (x == NULL) {
        post("[Python] py4pd capsule not found. The module pd must be used "
             "inside py4pd object or functions.");
        return NULL;
    }

    // check if there is no argument
    if (!PyArg_ParseTuple(args, "")) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.py4pdfolder: no argument expected");
        return NULL;
    }
    return PyUnicode_FromString(x->Py4pdPath->s_name);
}

// =================================
/**
 * @brief Returns the tmp folder for py4pd (normally in ~/.py4pd)
 */
static PyObject *Py4pdMod_GetPy4pdTmpFolder(PyObject *self, PyObject *args) {
    (void)self;
    if (!PyArg_ParseTuple(args, "")) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.samplerate: no argument expected");
        return NULL;
    }
    t_py *x = Py4pdUtils_GetObject(self);
    if (x == NULL) {
        post("[Python] py4pd capsule not found. The module pd must be used "
             "inside py4pd object or functions.");
        return NULL;
    }
    Py4pdUtils_CreateTempFolder(x);
    return PyUnicode_FromString(x->TempPath->s_name);
}

// Py4pdMod_GetPdSysPath

// =================================
/**
 * @brief Returns the tmp folder for py4pd (normally in ~/.py4pd)
 */
static PyObject *Py4pdMod_GetPdSearchPaths(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    PyObject *pdPathList = PyList_New(0);
    for (int i = 0; 1; i++) {
        char const *pathelem = namelist_get(STUFF->st_searchpath, i);
        if (!pathelem) {
            break;
        }
        // add pathelem to pdPathList
        PyList_Append(pdPathList, PyUnicode_FromString(pathelem));
    }
    return pdPathList;
}
// =================================
/**
 * @brief Shows an image using something similar to else/pic
 */
static PyObject *Py4pdMod_ShowImage(PyObject *self, PyObject *args) {
    (void)self;
    char *string;
    t_py *x = Py4pdUtils_GetObject(self);
    if (x == NULL) {
        PyErr_SetString(PyExc_TypeError, "[Python] py4pd capsule not found");
        return NULL;
    }
    if (PyArg_ParseTuple(args, "s", &string)) {
        t_symbol *filename = gensym(string);
        if (x->DefImg) {
            x->DefImg = 0;
        }
        if (access(filename->s_name, F_OK) == -1) {
            pd_error(x, "[Python] File %s not found.", filename->s_name);
            x->DefImg = 1;
            Py4pdPic_ErasePic(x, x->Glist);
            sys_vgui(".x%lx.c itemconfigure %lx_picture -image "
                     "PY4PD_IMAGE_{%p}\n",
                     x->Canvas, x, x);
            Py4pdPic_Draw(x, x->Glist, 1);
            Py_RETURN_NONE;
        }

        FILE *file;
        char *ext = strrchr(filename->s_name, '.');
        if (strcmp(ext, ".ppm") == 0) {
            char magic_number[3];
            int width, height, max_color_value;
            file = fopen(filename->s_name, "r");
            int result =
                fscanf(file, "%s\n%d %d\n%d\n", magic_number, &width, &height, &max_color_value);
            if (result != 4) {
                PyErr_SetString(PyExc_TypeError, "[Python] pd.showimage: error reading file");
                return NULL;
            }
            x->Width = width;
            x->Height = height;
            fclose(file);
        } else if (strcmp(ext, ".gif") == 0) {
            file = fopen(filename->s_name, "rb");
            fseek(file, 6, SEEK_SET);
            int resultWidth = fread(&x->Width, 2, 1, file);
            int resultHeight = fread(&x->Height, 2, 1, file);
            if (resultWidth != 1 || resultHeight != 1) {
                PyErr_SetString(PyExc_TypeError, "[Python] pd.showimage: error reading file");
                return NULL;
            }
            fclose(file);
        } else if (strcmp(ext, ".png") == 0) {
            file = fopen(filename->s_name, "rb");
            int width, height;
            fseek(file, 16, SEEK_SET);
            int resultWidth = fread(&width, 4, 1, file);
            int resultHeight = fread(&height, 4, 1, file);
            if (resultWidth != 1 || resultHeight != 1) {
                PyErr_SetString(PyExc_TypeError, "[Python] pd.showimage: error reading file");
                return NULL;
            }
            fclose(file);
            width = Py4pdUtils_Ntohl(width);
            height = Py4pdUtils_Ntohl(height);
            x->Width = width;
            x->Height = height;
        } else {
            PyErr_SetString(PyExc_TypeError, "pd.showimage: file format not supported, use "
                                             ".ppm, .gif, or .png");
            return NULL;
        }

        if (glist_isvisible(x->Glist) && gobj_shouldvis((t_gobj *)x, x->Glist)) {
            const char *fileNameOpen = Py4pdPic_Filepath(x, filename->s_name);
            if (access(fileNameOpen, F_OK) == -1) {
                pd_error(x, "[Python] pd.showimage: file not found");
                PyErr_SetString(PyExc_TypeError, "[Python] pd.showimage: file not found");
                return NULL;
            }
            if (fileNameOpen) {
                x->PicFilePath = gensym(fileNameOpen);
                if (x->DefImg) {
                    x->DefImg = 0;
                }

                if (glist_isvisible(x->Glist) && gobj_shouldvis((t_gobj *)x, x->Glist)) {
                    Py4pdPic_ErasePic(x, x->Glist);
                    sys_vgui("if {[info exists %lx_picname] == 0} {image create "
                             "photo %lx_picname -file \"%s\"\n set %lx_picname "
                             "1\n}\n",
                             x->PicFilePath, x->PicFilePath, fileNameOpen, x->PicFilePath);
                    Py4pdPic_Draw(x, x->Glist, 1);
                }
            } else {
                pd_error(x, "Error displaying image, file not found");
                PyErr_Clear();
                Py_RETURN_NONE;
            }
        } else {
            // No Gui Visible
            PyErr_Clear();
            Py_RETURN_NONE;
        }
    } else {
        pd_error(x, "pd.showimage received wrong arguments");
        PyErr_Clear();
        Py_RETURN_NONE;
    }
    PyErr_Clear();
    Py_RETURN_NONE;
}

// =================================
/**
 * @brief Get the current sample rate
 */
static PyObject *Py4pdMod_PdSampleRate(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    float sr = sys_getsr(); // call PureData's sys_getsr function to get the
    return PyFloat_FromDouble(sr);
}

// =================================
/**
 * @brief Get the current patch vector size
 */
static PyObject *Py4pdMod_PdVecSize(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    t_py *x = Py4pdUtils_GetObject(self);
    t_sample vector;
    vector = x->VectorSize; // this is the patch vector size
    return PyLong_FromLong(vector);
}

// =================================
/**
 * @brief Get the number of channels of the object
 */
static PyObject *Py4pdMod_ObjNChannels(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    t_py *x = Py4pdUtils_GetObject(self);
    int channels;
    channels = x->nChs;
    return PyLong_FromLong(channels);
}

// =================================
/**
 * @brief Get the current patch zoom
 */
static PyObject *Py4pdMod_PdZoom(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    t_py *x = Py4pdUtils_GetObject(self);
    int zoom;
    if (x->Canvas != NULL) {
        zoom = (int)x->Zoom;
    } else {
        pd_error(NULL, "pd.patchzoom: canvas not found");
        zoom = 1;
    }
    return PyLong_FromLong(zoom);
}

// =================================
/**
 * @brief Check if there is a GUI (mainly for testing purposes)
 */
static PyObject *Py4pdMod_PdHasGui(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    return PyLong_FromLong(sys_havegui());
}

// =================================
/**
 * @brief You can save values for an object with this function.
 */
static PyObject *Py4pdMod_PdKey(PyObject *self, PyObject *args) {
    // get values from Dict salved in x->param
    (void)self;
    char *key;
    if (!PyArg_ParseTuple(args, "s", &key)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.key: no argument expected");
        return NULL;
    }

    t_py *x = Py4pdUtils_GetObject(self);
    if (x == NULL) {
        return NULL;
    }

    if (x->pObjVarDict == NULL) {
        PyErr_Clear();
        Py_RETURN_NONE;
    }

    PyObject *value = PyDict_GetItemString(x->pObjVarDict, key);
    if (value == NULL) {
        PyErr_Clear();
        Py_RETURN_NONE;
    }
    return value;
}

// =================================
/**
 * @brief Returns a string with the address of the object (good for obj only
 * variables)
 */
static PyObject *Py4pdMod_GetObjPointer(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    t_py *x = Py4pdUtils_GetObject(self);
    if (x == NULL) {
        PyErr_SetString(PyExc_TypeError, "py4pd capsule not found. The module pd must be used "
                                         "inside py4pd object or functions.");
        return NULL;
    }

    return PyUnicode_FromFormat("%p", x);
}

// =================================
/**
 * @brief Add a thing to the player, you can use play to play it
 */
static PyObject *Py4pdMod_AddThingToPlay(PyObject *self, PyObject *args, PyObject *keywords) {
    (void)self;
    (void)keywords;

    float onset;
    PyObject *thingToPlay;
    t_py *x = Py4pdUtils_GetObject(self);

    if (!PyArg_ParseTuple(args, "fO", &onset, &thingToPlay)) {
        PyErr_SetString(PyExc_TypeError, "pd.add_to_player: wrong arguments, it should be: "
                                         "pd.add_to_player(onset, thing2Output)");
        return NULL;
    }
    Py4pdPlayer_PlayerInsertThing(x, (int)onset, Py_BuildValue("O", thingToPlay));
    Py_RETURN_NONE;
}

// =================================
/**
 * @brief Clear the player values
 */
static PyObject *Py4pdMod_ClearPlayer(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    t_py *x = Py4pdUtils_GetObject(self);
    Py4pdPlayer_Clear(x);
    Py_RETURN_NONE;
}

// =================================
// ========= MODULE INIT ===========
// =================================
// =================================
PyMethodDef PdMethods[] = {

    // PureData inside Python
    {"out", (PyCFunction)Py4pdMod_PdOut, METH_VARARGS | METH_KEYWORDS,
     "Output in out0 from PureData"},
    {"send", Py4pdMod_PdSend, METH_VARARGS,
     "Send message to PureData, it can be received with the object "
     "[receive]"},
    {"print", (PyCFunction)Py4pdMod_PdPrint, METH_VARARGS | METH_KEYWORDS,
     "Print informations in PureData Console"},
    {"logpost", Py4pdMod_PdLogPost, METH_VARARGS, "Print informations in PureData Console"},
    {"error", Py4pdMod_PdError, METH_VARARGS,
     "Print informations in error format (red) in PureData Console"},
    {"tabwrite", (PyCFunction)Py4pdMod_PdTabWrite, METH_VARARGS | METH_KEYWORDS,
     "Write data to PureData tables/arrays"},
    {"tabread", (PyCFunction)Py4pdMod_PdTabRead, METH_VARARGS | METH_KEYWORDS,
     "Read data from PureData tables/arrays"},

    // Pic
    {"show_image", Py4pdMod_ShowImage, METH_VARARGS,
     "Show image in PureData, it must be .gif, .bmp, .ppm"},

    // Files
    {"get_patch_dir", Py4pdMod_GetPatchHome, METH_VARARGS, "Get PureData Patch Path Folder"},
    {"get_py4pd_dir", Py4pdMod_GetPy4pdFolder, METH_VARARGS, "Get PureData Py4PD Folder"},
    {"get_temp_dir", Py4pdMod_GetPy4pdTmpFolder, METH_VARARGS, "Get PureData Temp Folder"},
    {"get_pd_search_paths", Py4pdMod_GetPdSearchPaths, METH_NOARGS, "Get PureData sys path"},

    // User
    {"get_key", Py4pdMod_PdKey, METH_VARARGS, "Get Object User Parameters"},

    // pd
    {"get_sample_rate", Py4pdMod_PdSampleRate, METH_NOARGS, "Get PureData SampleRate"},
    {"get_vec_size", Py4pdMod_PdVecSize, METH_NOARGS, "Get PureData Vector Size"},
    {"get_num_channels", Py4pdMod_ObjNChannels, METH_NOARGS,
     "Return the amount of channels in the object"},
    {"pd_has_gui", Py4pdMod_PdHasGui, METH_NOARGS, "Return True of False if pd has or no gui"},

    {"get_patch_zoom", Py4pdMod_PdZoom, METH_NOARGS, "Get Patch zoom"},
    {"get_outlet_count", Py4pdMod_PdGetOutletsCount, METH_NOARGS,
     "Get the Number of Outlets of one object."},
    {"get_obj_args", Py4pdMod_GetObjArgs, METH_NOARGS, "Returns list with all the args."},
    {"get_object_args", Py4pdMod_GetObjArgs, METH_NOARGS, "Returns list with all the args."},

    // library methods
    {"add_object", (PyCFunction)Py4pdLib_AddObj, METH_VARARGS | METH_KEYWORDS,
     "It adds python functions as objects"},

    // Others
    {"get_obj_pointer", Py4pdMod_GetObjPointer, METH_NOARGS, "Get PureData Object Pointer"},

    // Loops
    {"set_obj_var", Py4pdMod_SetObjVar, METH_VARARGS,
     "It sets a global variable for the Object, it is not clear after the "
     "execution of the function"},
    {"accum_obj_var", Py4pdMod_AccumObjVar, METH_VARARGS,
     "It adds the values in the end of the list"},
    {"get_obj_var", (PyCFunction)Py4pdMod_GetObjVar, METH_VARARGS | METH_KEYWORDS,
     "It gets a global variable for the Object, it is not clear after the "
     "execution of the function"},
    {"clear_obj_var", (PyCFunction)Py4pdMod_ClearObjVar, METH_VARARGS,
     "It clear the Dictionary of global variables"},

    {"get_py_type", (PyCFunction)Py4pdMod_GetPyObjectType, METH_VARARGS,
     "Returns the Python Object"},

    // player
    {"add_to_player", (PyCFunction)Py4pdMod_AddThingToPlay, METH_VARARGS | METH_KEYWORDS,
     "It adds a thing to the player"},
    {"clear_player", Py4pdMod_ClearPlayer, METH_NOARGS, "Remove all Python Objects of the player."},

    // Pip
    {"_pipinstall_requirements", Py4pdMod_PipInstallRequirements, METH_VARARGS | METH_KEYWORDS,
     "Install a package using pip"},

    // Internal
    {"_recursive", Py4pdMod_PdRecursiveCall, METH_VARARGS, "It calls a function recursively"},
    {"_open_patch", Py4pdMod_OpenPatch, METH_VARARGS, "Open a patch in PureData"},
    {NULL, NULL, 0, NULL} //
};

static PyObject *pdmodule_init(PyObject *self) {
    char OUT_string[MAXPDSTRING];
    pd_snprintf(OUT_string, sizeof(OUT_string), "py4pdOut_%p", self);

    char CLEAR_string[MAXPDSTRING];
    pd_snprintf(CLEAR_string, sizeof(CLEAR_string), "py4pdClear_%p", self);

    PyObject *puredata_samplerate, *puredata_vecsize, *visObject, *normalObject, *audioINObject,
        *audioOUTObject, *pdAudio, *Py4pd_OutLoopString, *Py4pd_ClearLoopString;

    puredata_samplerate = PyLong_FromLong(sys_getsr());
    puredata_vecsize = PyLong_FromLong(sys_getblksize());

    normalObject = PyLong_FromLong(PY4PD_NORMALOBJ);
    visObject = PyLong_FromLong(PY4PD_VISOBJ);
    audioINObject = PyLong_FromLong(PY4PD_AUDIOINOBJ);
    audioOUTObject = PyLong_FromLong(PY4PD_AUDIOOUTOBJ);
    pdAudio = PyLong_FromLong(PY4PD_AUDIOOBJ);

    Py4pd_OutLoopString = PyUnicode_FromString(OUT_string);
    Py4pd_ClearLoopString = PyUnicode_FromString(CLEAR_string);

    PyModule_AddObject(self, "SAMPLERATE", puredata_samplerate);
    PyModule_AddObject(self, "VECSIZE", puredata_vecsize);

    PyModule_AddObject(self, "NORMAL", normalObject);
    PyModule_AddObject(self, "VIS", visObject);
    PyModule_AddObject(self, "AUDIOIN", audioINObject);
    PyModule_AddObject(self, "AUDIOOUT", audioOUTObject);
    PyModule_AddObject(self, "AUDIO", pdAudio);

    PyModule_AddObject(self, "OUTLOOP", Py4pd_OutLoopString);
    PyModule_AddObject(self, "CLEARLOOP", Py4pd_ClearLoopString);

    // PD TYPES
    PyObject *pA_GIMME = PyLong_FromLong(A_GIMME);
    PyObject *pA_FLOAT = PyLong_FromLong(A_FLOAT);
    PyObject *pA_SYMBOL = PyLong_FromLong(A_SYMBOL);

    PyModule_AddObject(self, "A_GIMME", pA_GIMME);
    PyModule_AddObject(self, "A_FLOAT", pA_FLOAT);
    PyModule_AddObject(self, "A_SYMBOL", pA_SYMBOL);

    if (PyType_Ready(&Py4pdNewObj_Type) < 0)
        return NULL;
    Py_INCREF(&Py4pdNewObj_Type);

    PyModule_AddObject(self, "new_object", (PyObject *)&Py4pdNewObj_Type);

    return 0;
}

// =================================
static PyModuleDef_Slot pdmodule_slots[] = {
    {Py_mod_exec, pdmodule_init}, // Initialization phase
#if PYTHON_REQUIRED_VERSION(3, 12)
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
#endif
    {0, NULL} // End of slots
};

// =================================
static struct PyModuleDef pdModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pd", /* name of module */
    .m_doc = "pd module provide function to interact with PureData, see the "
             "docs in www.charlesneimog.github.io/py4pd",
    .m_size = 0,
    .m_methods = PdMethods, // Methods of the module
    .m_slots = pdmodule_slots,
};

// =================================
PyMODINIT_FUNC PyInit_pd() {
    import_array() PyObject *py4pdModule;
    py4pdModule = PyModuleDef_Init(&pdModule);
    if (py4pdModule == NULL) {
        return NULL;
    }

    return py4pdModule;
}

// ============================
