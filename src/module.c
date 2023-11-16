#include "py4pd.h"

#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#include <numpy/arrayobject.h>

// =================================
static pdcollectHash *CreatePdcollectHash(
    int size) { // TODO: make thing to free table when object is deleted?
    pdcollectHash *hash_table = (pdcollectHash *)malloc(sizeof(pdcollectHash));
    hash_table->size = size;
    hash_table->count = 0;
    hash_table->items = (pdcollectItem **)calloc(size, sizeof(pdcollectItem *));
    return hash_table;
}

// =================================
static unsigned int HashFunction(pdcollectHash *hash_table, char *key) {
    unsigned long hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % hash_table->size;
}

// =================================
static void InsertItem(pdcollectHash *hash_table, char *key,
                       PyObject *obj) { // TODO: make it return something
    unsigned int index = HashFunction(hash_table, key);
    pdcollectItem *item = hash_table->items[index];
    if (item == NULL && hash_table->count <= hash_table->size) {
        item = (pdcollectItem *)malloc(sizeof(pdcollectItem));
        item->key = strdup(key);
        item->pItem = obj;
        hash_table->items[index] = item;
        item->aCumulative = 0;
        hash_table
            ->count++; // TODO: Make resizeable table. For now we have 8 items.
        return;
    } else if (hash_table->count > hash_table->size) {
        PyErr_SetString(PyExc_MemoryError,
                        "[Python] pd.setglobalvar: Memory Error");
        return;
    } else if (item != NULL) {

        item->pItem = Py_BuildValue("O", obj);
        return;
    }
}

// =================================
static void AccumItem(pdcollectHash *hash_table, char *key, PyObject *obj) {
    unsigned int index = HashFunction(hash_table, key);
    pdcollectItem *item = hash_table->items[index];
    if (item == NULL && hash_table->count <= hash_table->size) {
        item = (pdcollectItem *)malloc(sizeof(pdcollectItem));
        item->key = strdup(key);
        item->pList = PyList_New(0);
        hash_table->items[index] = item;
    } else if (hash_table->count > hash_table->size) {
        PyErr_SetString(PyExc_MemoryError,
                        "[Python] pd.setglobalvar: memory error");
        return;
    }
    if (item->wasCleaned)
        item->wasCleaned = 0;
    PyList_Append(item->pList, obj);
    return;
}

// =================================
static void ClearItem(pdcollectHash *hash_table, char *key) {
    unsigned int index = HashFunction(hash_table, key);
    pdcollectItem *item = hash_table->items[index];
    if (item == NULL) {
        return;
    }
    if (item->wasCleaned)
        return;
    item->wasCleaned = 1;
    free(item->key);
    Py_DECREF(item->pItem);
    Py4pdUtils_MemLeakCheck(item->pItem, 0, "ClearItem");
    free(item);
    hash_table->items[index] = NULL;
}

// =================================
static void ClearList(pdcollectHash *hash_table, char *key) {

    unsigned int index = HashFunction(hash_table, key);
    pdcollectItem *item = hash_table->items[index];
    if (item == NULL) {
        return;
    }
    if (item->wasCleaned) {
        return;
    }
    item->wasCleaned = 1;
    free(item->key);
    Py_DECREF(item->pList);
    Py4pdUtils_MemLeakCheck(item->pList, 0, "pList");
    free(item);
    hash_table->items[index] = NULL;
}

// =================================
static pdcollectItem *GetObjArr(pdcollectHash *hash_table, char *key) {
    unsigned int index = HashFunction(hash_table, key);
    pdcollectItem *item = hash_table->items[index];
    if (item == NULL)
        return NULL;
    return item;
}

// =================================
static void FreePdcollectItem(pdcollectItem *item) {
    if (item == NULL) {
        return;
    }
    if (item->wasCleaned) {
        return;
    }
    item->wasCleaned = 1;
    free(item->key);

    // Free the appropriate object, depending on whether it's a single item or a
    // list
    if (item->pList) {
        Py_DECREF(item->pList);
        Py4pdUtils_MemLeakCheck(item->pList, 0, "pList");
    } else if (item->pItem) {
        Py_DECREF(item->pItem);
    }
    free(item);
}

// =================================
void FreePdcollectHash(pdcollectHash *hash_table) {
    if (hash_table == NULL) {
        return;
    }
    for (int i = 0; i < hash_table->size; ++i) {
        FreePdcollectItem(hash_table->items[i]);
    }
    free(hash_table->items);
    free(hash_table);
}

// =================================
// =================================
// =================================
static PyObject *Py4pdMod_SetGlobalVar(PyObject *self, PyObject *args) {
    (void)self;
    t_py *py4pd = Py4pdUtils_GetObject(self);
    if (py4pd == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "[Python] pd.setglobalvar: py4pd is NULL");
        return NULL;
    }
    char *key;
    char *varName;
    PyObject *pValueScript;
    if (!PyArg_ParseTuple(args, "sO", &varName, &pValueScript)) {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.setglobalvar: wrong arguments");
        return NULL;
    }

    key = malloc(strlen(varName) + 40);
    snprintf(key, strlen(varName) + 40, "%s_%p", varName, py4pd);
    if (py4pd->pdcollect == NULL) {
        py4pd->pdcollect = CreatePdcollectHash(8);
    }
    InsertItem(py4pd->pdcollect, key, pValueScript);
    free(key);
    Py_RETURN_TRUE;
}

// =================================
static PyObject *Py4pdMod_GetGlobalVar(PyObject *self, PyObject *args,
                                       PyObject *keywords) {
    (void)self;

    t_py *py4pd = Py4pdUtils_GetObject(self);
    if (py4pd == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "[Python] pd.setglobalvar: py4pd is NULL");
        return NULL;
    }
    char *key;
    char *varName;

    if (!PyArg_ParseTuple(args, "s", &varName)) {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.setglobalvar: wrong arguments");
        return NULL;
    }

    key = malloc(strlen(varName) + 40);
    snprintf(key, strlen(varName) + 40, "%s_%p", varName, py4pd);
    if (py4pd->pdcollect == NULL) {
        py4pd->pdcollect = CreatePdcollectHash(8); // TODO: add way to resize
    }

    pdcollectItem *item = GetObjArr(py4pd->pdcollect, key);

    if (keywords != NULL) {
        PyObject *pString = PyUnicode_FromString("initial_value");
        PyObject *pValueInit = PyDict_GetItem(keywords, pString);
        Py_DECREF(pString);
        if (pValueInit != NULL && item == NULL) {
            InsertItem(py4pd->pdcollect, key, pValueInit);
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
        return item->pList;
    } else {
        return item->pItem;
    }
}

// =================================
static PyObject *Py4pdMod_AccumGlobalVar(PyObject *self, PyObject *args) {

    (void)self;

    t_py *py4pd = Py4pdUtils_GetObject(self);
    if (py4pd == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "[Python] pd.setglobalvar: py4pd is NULL");
        return NULL;
    }
    char *key;
    char *varName;
    PyObject *pValueScript;
    if (!PyArg_ParseTuple(args, "sO", &varName, &pValueScript)) {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.setglobalvar: wrong arguments");
        return NULL;
    }
    key = malloc(strlen(varName) + 40);
    snprintf(key, strlen(varName) + 40, "%s_%p", varName, py4pd);
    if (py4pd->pdcollect == NULL) {
        py4pd->pdcollect = CreatePdcollectHash(8);
    }
    AccumItem(py4pd->pdcollect, key, pValueScript);
    py4pd->pdcollect->items[HashFunction(py4pd->pdcollect, key)]->aCumulative =
        1;
    free(key);

    Py_RETURN_TRUE;
}

// =================================
static PyObject *Py4pdMod_ClearGlobalVar(PyObject *self, PyObject *args) {
    (void)self;

    t_py *py4pd = Py4pdUtils_GetObject(self);
    if (py4pd == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "[Python] pd.setglobalvar: py4pd is NULL");
        return NULL;
    }
    char *key;
    char *varName;

    if (!PyArg_ParseTuple(args, "s", &varName)) {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.setglobalvar: wrong arguments");
        return NULL;
    }

    key = malloc(strlen(varName) + 40);
    snprintf(key, strlen(varName) + 40, "%s_%p", varName, py4pd);
    if (py4pd->pdcollect == NULL) {
        free(key);
        Py_RETURN_TRUE;
    }
    pdcollectItem *objArr = GetObjArr(py4pd->pdcollect, key);
    if (objArr != NULL) {
        if (objArr->wasCleaned) {
            free(key);
            Py_RETURN_TRUE;
        }
    } else {
        free(key);
        Py_RETURN_TRUE;
    }

    if (objArr->aCumulative)
        ClearList(py4pd->pdcollect, key);
    else
        ClearItem(py4pd->pdcollect, key);

    free(key);
    Py_RETURN_TRUE;
}

// ======================================
static PyObject *Py4pdMod_GetObjArgs(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    t_py *py4pd = Py4pdUtils_GetObject(self);
    if (py4pd == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "[Python] pd.setglobalvar: py4pd is NULL");
        return NULL;
    }

    PyObject *pList = PyList_New(0);
    for (int i = 0; i < py4pd->objArgsCount; i++) {
        if (py4pd->pdObjArgs[i].a_type == A_FLOAT) {
            int isInt = (int)py4pd->pdObjArgs[i].a_w.w_float ==
                        py4pd->pdObjArgs[i].a_w.w_float;
            if (isInt) {
                PyObject *Number =
                    PyLong_FromLong(py4pd->pdObjArgs[i].a_w.w_float);
                PyList_Append(pList, Number);
                Py_DECREF(Number);
            } else {
                PyObject *Number =
                    PyFloat_FromDouble(py4pd->pdObjArgs[i].a_w.w_float);
                PyList_Append(pList, Number);
                Py_DECREF(Number);
            }

        } else if (py4pd->pdObjArgs[i].a_type == A_SYMBOL) {
            PyObject *strObj =
                PyUnicode_FromString(py4pd->pdObjArgs[i].a_w.w_symbol->s_name);
            PyList_Append(pList, strObj);
            Py_DECREF(strObj);
        } else {
            // post("In pd.getobjargs: unknown type");
        }
    }
    return pList;
}

// ======================================
static void Py4pdMod_RecursiveTick(t_py *x) {
    t_py4pd_pValue *pdPyValue =
        (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
    pdPyValue->pValue = x->recursiveObject;
    pdPyValue->objectsUsing = 0;
    Py4pdUtils_ConvertToPd(x, pdPyValue, x->out1);
    Py_DECREF(pdPyValue->pValue); // delete thing
    free(pdPyValue);
    Py_LeaveRecursiveCall();
}

// ======================================
static PyObject *Py4pdMod_PdRecursiveCall(PyObject *self, PyObject *args) {

    (void)self;

    PyObject *pValue;
    if (!PyArg_ParseTuple(args, "O", &pValue)) {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.setglobalvar: wrong arguments");
        return NULL;
    }

    t_py *py4pd = Py4pdUtils_GetObject(self);
    py4pd->recursiveCalls++;

    Py_EnterRecursiveCall("[py4pd] Exceeded maximum recursion depth");
    if (py4pd->recursiveCalls ==
        py4pd->stackLimit) { // seems to be the limit in PureData,
        py4pd->recursiveObject = pValue;
        if (py4pd->recursiveClock == NULL)
            py4pd->recursiveClock =
                clock_new(py4pd, (t_method)Py4pdMod_RecursiveTick);
        Py_INCREF(pValue); // avoid thing to be deleted
        py4pd->recursiveCalls = 0;
        clock_delay(py4pd->recursiveClock, 0);
        Py_RETURN_TRUE;
    }

    if (py4pd == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "[pd._recursive] pd.recursive: py4pd is NULL");
        return NULL;
    }

    t_py4pd_pValue *pdPyValue =
        (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
    pdPyValue->pValue = pValue;
    pdPyValue->objectsUsing = 0;
    Py4pdUtils_ConvertToPd(py4pd, pdPyValue, py4pd->out1);
    free(pdPyValue);
    Py_LeaveRecursiveCall();
    py4pd->recursiveCalls = 0;
    Py_RETURN_TRUE;
}

// ======================================
// ======== py4pd embbeded module =======
// ======================================
static PyObject *Py4pdMod_PdGetOutCount(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    t_py *py4pd = Py4pdUtils_GetObject(self);
    if (py4pd == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "[Python] py4pd capsule not found. The module pd must "
                        "be used inside py4pd object or functions.");
        return NULL;
    }
    return PyLong_FromLong(py4pd->outAUX->u_outletNumber);
}

// =================================
static PyObject *Py4pdMod_PdOut(PyObject *self, PyObject *args,
                                PyObject *keywords) {

    (void)self;

    t_py *py4pd = Py4pdUtils_GetObject(self);
    if (py4pd == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "[Python] py4pd capsule not found. The module pd must "
                        "be used inside py4pd object or functions.");
        return NULL;
    }

    PyObject *pValue;

    if (!PyArg_ParseTuple(args, "O", &pValue)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.out: wrong arguments");
        return NULL;
    }

    t_py4pd_pValue *pdPyValue =
        (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
    pdPyValue->pValue = pValue;
    pdPyValue->objectsUsing = 0;
    pdPyValue->pdout = 1;

    if (keywords != NULL && py4pd->outAUX != NULL) {
        PyObject *outletNumber = PyDict_GetItemString(
            keywords, "out_n"); // it gets the data type output
        if (outletNumber == NULL) {
            Py4pdUtils_ConvertToPd(py4pd, pdPyValue, py4pd->out1);
            free(pdPyValue);
            Py_RETURN_TRUE;
        }

        if (!PyLong_Check(outletNumber)) {
            PyErr_SetString(PyExc_TypeError,
                            "[Python] pd.out: out_n must be an integer.");
            Py_DECREF(pValue);
            return NULL;
        }
        int outletNumberInt = PyLong_AsLong(outletNumber);
        if (outletNumberInt == 0) {
            Py4pdUtils_ConvertToPd(py4pd, pdPyValue, py4pd->out1);
            free(pdPyValue);
            Py_RETURN_TRUE;
        } else {
            outletNumberInt--;
            if ((py4pd->outAUX->u_outletNumber > 0) &&
                (outletNumberInt < py4pd->outAUX->u_outletNumber)) {
                Py4pdUtils_ConvertToPd(py4pd, pdPyValue,
                                       py4pd->outAUX[outletNumberInt].u_outlet);
                free(pdPyValue);
                Py_RETURN_TRUE;
            } else {
                PyErr_SetString(
                    PyExc_TypeError,
                    "[Python] pd.out: Please check the number of outlets.");
                return NULL;
            }
        }
    } else {
        Py4pdUtils_ConvertToPd(py4pd, pdPyValue, py4pd->out1);
        free(pdPyValue);
    }
    Py_RETURN_TRUE;
}
// =================================
static PyObject *Py4pdMod_PdPrint(PyObject *self, PyObject *args,
                                  PyObject *keywords) {
    (void)self;
    int printPrefix = 1;
    int objPrefix = 1;

    t_py *py4pd = Py4pdUtils_GetObject(self);
    if (py4pd == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "[Python] py4pd capsule not found. The module pd must "
                        "be used inside py4pd object or functions.");
        return NULL;
    }

    if (keywords == NULL) {
        printPrefix = 1;
    } else {
        printPrefix =
            PyDict_Contains(keywords, PyUnicode_FromString("show_prefix"));
        objPrefix =
            PyDict_Contains(keywords, PyUnicode_FromString("obj_prefix"));
        if (printPrefix == -1)
            pd_error(NULL, "[Python] pd.print: error in show_prefix argument.");
        else if (printPrefix == 1) {
            PyObject *resize_value =
                PyDict_GetItemString(keywords, "show_prefix");
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
            PyObject *resize_value =
                PyDict_GetItemString(keywords, "obj_prefix");
            if (resize_value == Py_True)
                objPrefix = 1;
            else if (resize_value == Py_False)
                objPrefix = 0;
            else {
                post("[Python] pd.print: obj_prefix argument must be True or "
                     "False.");
                objPrefix = 0;
            }
        } else
            objPrefix = 0;
    }

    PyObject *obj;
    if (PyArg_ParseTuple(args, "O", &obj)) {
        PyObject *str = PyObject_Str(obj);
        if (str == NULL) {
            Py_DECREF(str);
            PyErr_SetString(
                PyExc_TypeError,
                "[Python] pd.print failed to convert object to string.");
            return NULL;
        }
        const char *str_value = PyUnicode_AsUTF8(str);
        if (str_value == NULL) {
            PyErr_SetString(
                PyExc_TypeError,
                "[Python] pd.print failed to convert string object to UTF-8.");
            Py_DECREF(str);
            return NULL;
        }
        if (printPrefix == 1) { //
            if (py4pd->objectName == NULL) {
                post("[Python]: %s", str_value);
            } else {
                post("[%s]: %s", py4pd->objectName->s_name, str_value);
            }
            Py_RETURN_TRUE;
        } else {
            post("%s", str_value);
        }
        Py_DECREF(str);
    } else {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.print works with strings, numbers, and "
                        "any other valid Python object.");
        Py_DECREF(obj);
        return NULL;
    }
    Py_DECREF(obj);
    Py_RETURN_TRUE;
}

// =================================
static PyObject *Py4pdMod_PdLogPost(PyObject *self, PyObject *args) {
    (void)self;
    int postlevel;
    char *string;

    if (PyArg_ParseTuple(args, "is", &postlevel, &string)) {
        logpost(NULL, postlevel, "%s", string);
        return PyLong_FromLong(0);
    } else {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.logpost works with strings and numbers.");
        return NULL;
    }
}

// =================================
static PyObject *Py4pdMod_PdError(PyObject *self, PyObject *args) {
    (void)self;

    char *string;

    t_py *py4pd = Py4pdUtils_GetObject(self);

    if (py4pd->objType > PY4PD_VISOBJ) { // if audio object
        py4pd->audioError = 1;
    }

    if (PyArg_ParseTuple(args, "s", &string)) {
        if (py4pd == NULL) {
            pd_error(NULL, "%s", string);
            PyErr_Clear();
            return PyLong_FromLong(0);
        }

        if (py4pd->pyObject == 1) {
            pd_error(py4pd, "[%s]: %s", py4pd->objectName->s_name, string);
        } else {
            if (py4pd->function_name == NULL) {
                pd_error(py4pd, "%s", string);
            } else {
                pd_error(py4pd, "[%s]: %s", py4pd->function_name->s_name,
                         string);
            }
        }
        PyErr_Clear();
    } else {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] argument of pd.error must be a string");
        return NULL;
    }
    return PyLong_FromLong(0);
}

// =================================
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
            post("[Python] pd.send not found object [r %s] in pd patch",
                 receiver);
        }
    } else if (PyArg_ParseTuple(args, "sf", &receiver, &floatNumber)) {
        t_symbol *symbol = gensym(receiver);
        if (symbol->s_thing) {
            pd_float(symbol->s_thing, floatNumber);
        } else {
            post("[Python] pd.send not found object [r %s] in pd patch",
                 receiver);
        }
    } else if (PyArg_ParseTuple(args, "si", &receiver, &intNumber)) {
        t_symbol *symbol = gensym(receiver);
        if (symbol->s_thing) {
            pd_float(symbol->s_thing, intNumber);
        } else {
            post("[Python] pd.send not found object [r %s] in pd patch",
                 receiver);
        }
    } else if (PyArg_ParseTuple(args, "sO", &receiver, &listargs)) {
        if (PyDict_Check(listargs)) {
            char error_message[100];
            sprintf(
                error_message,
                "[Python] pd.send received a type 'dict', it must be a list, "
                "string, int, or float.");
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
                list_array[i].a_type = A_FLOAT;
                list_array[i].a_w.w_float = result_float;
            } else if (PyFloat_Check(pValue_i)) {

                double result = PyFloat_AsDouble(pValue_i);
                float result_float = (float)result;
                list_array[i].a_type = A_FLOAT;
                list_array[i].a_w.w_float = result_float;
            } else if (PyUnicode_Check(pValue_i)) {
                const char *result = PyUnicode_AsUTF8(pValue_i);
                list_array[i].a_type = A_SYMBOL;
                list_array[i].a_w.w_symbol = gensym(result);
            } else if (Py_IsNone(pValue_i)) {
                // post("None");
            } else {
                char error_message[100];
                sprintf(error_message,
                        "[Python] received a type '%s' in index %d of the "
                        "list, it must be a string, int, or float.",
                        pValue_i->ob_type->tp_name, i);
                PyErr_SetString(PyExc_TypeError, error_message);
                Py_DECREF(pValue_i);
                Py_DECREF(args);
                free(list_array);
                return NULL;
            }
            // Py_DECREF(pValue_i);
        }
        if (gensym(receiver)->s_thing) {
            pd_list(gensym(receiver)->s_thing, &s_list, list_size, list_array);
        } else {
            pd_error(NULL, "[Python] object [r %s] not found", receiver);
        }
    } else {
        char error_message[100];
        PyObject *pValue_i = PyTuple_GetItem(args, 1);
        sprintf(error_message,
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
static PyObject *Py4pdMod_PdTabWrite(PyObject *self, PyObject *args,
                                     PyObject *keywords) {
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
    t_py *py4pd = Py4pdUtils_GetObject(self);
    if (py4pd == NULL) {
        post("[Python] py4pd capsule not found. The module pd must be used "
             "inside py4pd object or functions.");
        return NULL;
    }

    if (PyArg_ParseTuple(args, "sO", &string, &samples)) {
        t_symbol *pd_symbol = gensym(string);
        if (!(pdarray = (t_garray *)pd_findbyclass(pd_symbol, garray_class))) {
            pd_error(py4pd, "[Python] Array %s not found.", string);
            return NULL;
        } else if (!garray_getfloatwords(pdarray, &vecsize, &vec)) {
            pd_error(py4pd, "[Python] Bad template for tabwrite '%s'.", string);
            return NULL;
        } else {
            int i;
            _import_array();
            if (samples == NULL) {
                pd_error(py4pd, "[Python] pd.tabwrite: wrong arguments");
                return NULL;
            }
            if (PyList_Check(samples)) {
                if (resize == 1) {
                    garray_resize_long(pdarray, PyList_Size(samples));
                    vecsize = PyList_Size(samples);
                    garray_getfloatwords(pdarray, &vecsize, &vec);
                }
                for (i = 0; i < vecsize; i++) {
                    float result_float =
                        (float)PyFloat_AsDouble(PyList_GetItem(samples, i));
                    vec[i].w_float = result_float;
                }
            } else if (PyArray_Check(samples)) {
                PyArrayObject *pArray =
                    PyArray_GETCONTIGUOUS((PyArrayObject *)samples);
                PyArray_Descr *pArrayType =
                    PyArray_DESCR(pArray); // double or float
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
                             py4pd->objectName->s_name, pArrayType->type_num);
                    py4pd->audioError = 1;
                }

            }

            else {
                pd_error(py4pd, "[Python] pd.tabwrite: wrong arguments");
                return NULL;
            }
            garray_redraw(pdarray);
            PyErr_Clear();
        }
    }
    Py_RETURN_TRUE;
}

// =================================
static PyObject *Py4pdMod_PdTabRead(PyObject *self, PyObject *args,
                                    PyObject *keywords) {
    (void)self;
    int vecsize;
    t_garray *pdarray;
    t_word *vec;
    char *string;
    int numpy;

    // ================================
    t_py *py4pd = Py4pdUtils_GetObject(self);
    // ================================

    if (keywords == NULL) {
        numpy = 0;
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
                int numpyArrayImported = _import_array();
                if (numpyArrayImported == 0) {
                    py4pd->numpyImported = 1;
                } else {
                    py4pd->numpyImported = 0;
                    pd_error(py4pd,
                             "[py4pd] Not possible to import numpy array");
                    return NULL;
                }
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
            pd_error(py4pd, "[Python] Array %s not found.", string);
            PyErr_SetString(PyExc_TypeError,
                            "[Python] pd.tabread: array not found");
        } else {
            int i;
            if (numpy == 0) {
                garray_getfloatwords(pdarray, &vecsize, &vec);
                PyObject *list = PyList_New(vecsize);
                for (i = 0; i < vecsize; i++) {
                    PyList_SetItem(list, i, PyFloat_FromDouble(vec[i].w_float));
                }
                PyErr_Clear();
                return list;
            } else if (numpy == 1) {
                garray_getfloatwords(pdarray, &vecsize, &vec);
                const npy_intp dims = vecsize;
                // send double float array to numpy
                PyObject *array =
                    PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, vec);
                PyErr_Clear();
                return array;

            } else {
                pd_error(py4pd, "[Python] Check the keyword arguments.");
                return NULL;
            }
        }
    } else {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.tabread: wrong arguments");
        return NULL;
    }
    return NULL;
}

// =================================
static PyObject *Py4pdMod_GetPatchHome(PyObject *self, PyObject *args) {
    (void)self;

    t_py *py4pd = Py4pdUtils_GetObject(self);
    if (py4pd == NULL) {
        post("[Python] py4pd capsule not found. The module pd must be used "
             "inside py4pd object or functions.");
        return NULL;
    }

    // check if there is no argument
    if (!PyArg_ParseTuple(args, "")) {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.home: no argument expected");
        return NULL;
    }
    return PyUnicode_FromString(py4pd->pdPatchFolder->s_name);
}

// =================================
static PyObject *Py4pdMod_GetObjFolder(PyObject *self, PyObject *args) {
    (void)self;

    t_py *py4pd = Py4pdUtils_GetObject(self);
    if (py4pd == NULL) {
        post("[Python] py4pd capsule not found. The module pd must be used "
             "inside py4pd object or functions.");
        return NULL;
    }

    // check if there is no argument
    if (!PyArg_ParseTuple(args, "")) {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.py4pdfolder: no argument expected");
        return NULL;
    }
    return PyUnicode_FromString(py4pd->py4pdPath->s_name);
}

// =================================
static PyObject *Py4pdMod_GetObjTmpFolder(PyObject *self, PyObject *args) {
    (void)self;
    if (!PyArg_ParseTuple(args, "")) {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.samplerate: no argument expected");
        return NULL;
    }
    t_py *py4pd = Py4pdUtils_GetObject(self);
    if (py4pd == NULL) {
        post("[Python] py4pd capsule not found. The module pd must be used "
             "inside py4pd object or functions.");
        return NULL;
    }
    Py4pdUtils_CreateTempFolder(py4pd);
    return PyUnicode_FromString(py4pd->tempPath->s_name);
}

// =================================
static PyObject *Py4pdMod_ShowImage(PyObject *self, PyObject *args) {
    (void)self;
    char *string;
    t_py *py4pd = Py4pdUtils_GetObject(self);
    if (py4pd == NULL) {
        PyErr_SetString(PyExc_TypeError, "[Python] py4pd capsule not found");
        return NULL;
    }
    if (PyArg_ParseTuple(args, "s", &string)) {
        t_symbol *filename = gensym(string);
        if (py4pd->x_def_img) {
            py4pd->x_def_img = 0;
        }
        if (access(filename->s_name, F_OK) == -1) {
            pd_error(py4pd, "[Python] File %s not found.", filename->s_name);

            // reset image to default
            py4pd->x_def_img = 1;
            // py4pd->x_width = 250;
            // py4pd->x_height = 250;
            Py4pdPic_Erase(py4pd, py4pd->glist);
            sys_vgui(".x%lx.c itemconfigure %lx_picture -image "
                     "PY4PD_IMAGE_{%p}\n",
                     py4pd->canvas, py4pd, py4pd);
            Py4pdPic_Draw(py4pd, py4pd->glist, 1);
            Py_RETURN_NONE;
        }

        FILE *file;
        char *ext = strrchr(filename->s_name, '.');
        if (strcmp(ext, ".ppm") == 0) {
            char magic_number[3];
            int width, height, max_color_value;
            file = fopen(filename->s_name, "r");
            fscanf(file, "%s\n%d %d\n%d\n", magic_number, &width, &height,
                   &max_color_value);
            py4pd->x_width = width;
            py4pd->x_height = height;
            fclose(file);
        } else if (strcmp(ext, ".gif") == 0) {
            file = fopen(filename->s_name, "rb");
            fseek(file, 6, SEEK_SET);
            fread(&py4pd->x_width, 2, 1, file);
            fread(&py4pd->x_height, 2, 1, file);
            fclose(file);
        } else if (strcmp(ext, ".png") == 0) {
            file = fopen(filename->s_name, "rb");
            int width, height;
            fseek(file, 16, SEEK_SET);
            fread(&width, 4, 1, file);
            fread(&height, 4, 1, file);
            fclose(file);
            width = Py4pdUtils_Ntohl(width);
            height = Py4pdUtils_Ntohl(height);
            py4pd->x_width = width;
            py4pd->x_height = height;
        }

        else {
            pd_error(py4pd, "[Python] pd.showimage: file format not supported");
            PyErr_SetString(PyExc_TypeError,
                            "[Python] pd.showimage: file format not supported");
            return NULL;
        }

        if (glist_isvisible(py4pd->glist) &&
            gobj_shouldvis((t_gobj *)py4pd, py4pd->glist)) {
            const char *file_name_open =
                Py4pdPic_Filepath(py4pd, filename->s_name);
            if (access(file_name_open, F_OK) == -1) {
                pd_error(py4pd, "[Python] pd.showimage: file not found");
                PyErr_SetString(PyExc_TypeError,
                                "[Python] pd.showimage: file not found");
                return NULL;
            }
            if (file_name_open) {
                py4pd->x_filename = filename;
                py4pd->x_fullname = gensym(file_name_open);

                if (py4pd->x_def_img) {
                    py4pd->x_def_img = 0;
                }

                if (glist_isvisible(py4pd->glist) &&
                    gobj_shouldvis((t_gobj *)py4pd, py4pd->glist)) {
                    Py4pdPic_Erase(py4pd, py4pd->glist);
                    sys_vgui(
                        "if {[info exists %lx_picname] == 0} {image create "
                        "photo %lx_picname -file \"%s\"\n set %lx_picname "
                        "1\n}\n",
                        py4pd->x_fullname, py4pd->x_fullname, file_name_open,
                        py4pd->x_fullname);
                    Py4pdPic_Draw(py4pd, py4pd->glist, 1);
                }
            } else {
                pd_error(py4pd, "Error displaying image, file not found");
                PyErr_Clear();
                Py_RETURN_NONE;
            }
        } else {
            pd_error(py4pd, "Error displaying image");
            PyErr_Clear();
            Py_RETURN_NONE;
        }
    } else {
        pd_error(py4pd, "pd.showimage received wrong arguments");
        PyErr_Clear();
        Py_RETURN_NONE;
    }
    PyErr_Clear();
    Py_RETURN_NONE;
}

// =================================
// ========== AUDIO CONFIG =========
// =================================

static PyObject *Py4pdMod_PdSampleRate(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    float sr = sys_getsr(); // call PureData's sys_getsr function to get the
                            // current sample rate
    return PyFloat_FromDouble(
        sr); // return the sample rate as a Python float object
}

// =================================
static PyObject *Py4pdMod_PdVecSize(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    t_py *py4pd = Py4pdUtils_GetObject(self);
    t_sample vector;
    vector = py4pd->vectorSize; // this is the patch vector size
    // TODO: Should I add someway to get the puredata's vector size? with
    // sys_getblksize()?

    return PyLong_FromLong(vector);
}

// =================================
static PyObject *Py4pdMod_ObjNChannels(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    t_py *py4pd = Py4pdUtils_GetObject(self);
    int channels;
    channels = py4pd->n_channels;
    // TODO: Should I add someway to get the puredata's vector size? with
    // sys_getblksize()?

    return PyLong_FromLong(channels);
}

// =================================
static PyObject *Py4pdMod_PdZoom(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    t_py *py4pd = Py4pdUtils_GetObject(self);
    int zoom;
    if (py4pd->canvas != NULL) {
        zoom = (int)py4pd->canvas->gl_zoom;
    } else {
        pd_error(NULL, "pd.patchzoom: canvas not found");
        zoom = 1;
    }
    return PyLong_FromLong(zoom);
}

// =================================
static PyObject *Py4pdMod_PdHasGui(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    return PyLong_FromLong(sys_havegui());
}

// =================================
// ========== Utilities ============
// =================================

static PyObject *Py4pdMod_PdKey(PyObject *self, PyObject *args) {
    // get values from Dict salved in x->param
    (void)self;
    char *key;
    if (!PyArg_ParseTuple(args, "s", &key)) {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.key: no argument expected");
        return NULL;
    }

    t_py *py4pd = Py4pdUtils_GetObject(self);
    if (py4pd == NULL) {
        return NULL;
    }

    if (py4pd->Dict == NULL) {
        PyErr_Clear();
        Py_RETURN_NONE;
    }

    PyObject *value = PyDict_GetItemString(py4pd->Dict, key);
    if (value == NULL) {
        PyErr_Clear();
        Py_RETURN_NONE;
    }
    return value;
}

// =================================
static PyObject *Py4pdMod_GetObjPointer(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    t_py *py4pd = Py4pdUtils_GetObject(self);
    if (py4pd == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "py4pd capsule not found. The module pd must be used "
                        "inside py4pd object or functions.");
        return NULL;
    }

    return PyUnicode_FromFormat("%p", py4pd);
}

// =================================
// ============ PLAYER =============
// =================================
static PyObject *Py4pdMod_AddThingToPlay(PyObject *self, PyObject *args,
                                         PyObject *keywords) {
    (void)self;
    (void)keywords;

    int onset;
    PyObject *thingToPlay;
    t_py *py4pd = Py4pdUtils_GetObject(self);

    if (!PyArg_ParseTuple(args, "iO", &onset, &thingToPlay)) {
        PyErr_SetString(PyExc_TypeError,
                        "pd.add2play: wrong arguments, it should be: "
                        "pd.add2play(onset, thingToPlay)");
        return NULL;
    }
    Py4pdLib_PlayerInsertThing(py4pd, onset, Py_BuildValue("O", thingToPlay));
    Py_RETURN_NONE;
}

// =================================
static PyObject *Py4pdMod_ClearPlayer(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    t_py *py4pd = Py4pdUtils_GetObject(self);
    Py4pdLib_Clear(py4pd);
    Py_RETURN_NONE;
}

// =================================
// ============= PIP ===============
// =================================
static PyObject *Py4pdMod_PipInstall(PyObject *self, PyObject *args) {
    (void)self;
    char *pipPackage;
    char *localORglobal;

    t_py *x = Py4pdUtils_GetObject(self);

    PyErr_Clear(); // this is probably called after an error, so we clear it

    if (!PyArg_ParseTuple(args, "ss", &localORglobal, &pipPackage)) {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.pipInstall: wrong arguments");
        return NULL;
    }

    PyObject *py4pdModule = PyImport_ImportModule("py4pd");
    if (py4pdModule == NULL) {
        pd_error(x, "[Python] pipInstall: py4pd module not found");
        return NULL;
    }
    PyObject *pipInstallFunction =
        PyObject_GetAttrString(py4pdModule, "pipinstall");
    if (pipInstallFunction == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "[Python] pd.pipInstall: pipinstall function not found");
        return NULL;
    }
    x->function = pipInstallFunction;

    PyObject *argsList = PyList_New(2);
    PyList_SetItem(argsList, 0, Py_BuildValue("s", localORglobal));
    PyList_SetItem(argsList, 1, Py_BuildValue("s", pipPackage));
    PyObject *argTuple = PyTuple_New(1);
    PyTuple_SetItem(argTuple, 0, argsList);

    t_py *prev_obj = NULL;
    int prev_obj_exists = 0;
    PyObject *MainModule = PyImport_ImportModule("pd");
    PyObject *oldObjectCapsule;

    if (MainModule != NULL) {
        oldObjectCapsule =
            PyDict_GetItemString(MainModule, "py4pd"); // borrowed reference
        if (oldObjectCapsule != NULL) {
            PyObject *py4pd_capsule =
                PyObject_GetAttrString(MainModule, "py4pd");
            prev_obj = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
            prev_obj_exists = 1;
        } else {
            prev_obj_exists = 0;
        }
    }

    PyObject *objectCapsule = Py4pdUtils_AddPdObject(x);

    if (objectCapsule == NULL) {
        pd_error(x, "[Python] Failed to add object to Python");
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyObject *ptype_str = PyObject_Str(ptype);
        PyObject *pvalue_str = PyObject_Str(pvalue);
        PyObject *ptraceback_str = PyObject_Str(ptraceback);
        const char *ptype_c = PyUnicode_AsUTF8(ptype_str);
        const char *pvalue_c = PyUnicode_AsUTF8(pvalue_str);
        const char *ptraceback_c = PyUnicode_AsUTF8(ptraceback_str);
        pd_error(x, "[Python] %s: %s\n%s", ptype_c, pvalue_c, ptraceback_c);
        return NULL;
    }

    PyObject *pValue = PyObject_CallObject(pipInstallFunction, argTuple);

    if (prev_obj_exists == 1 && pValue != NULL) {
        objectCapsule = Py4pdUtils_AddPdObject(prev_obj);
        if (objectCapsule == NULL) {
            pd_error(x, "[Python] Failed to add object to Python");
            return NULL;
        }
    }

    if (pValue == NULL) {
        pd_error(x, "[Python] pipInstall: pipinstall function failed");
        return NULL;
    } else {
        pd_error(x, "[Python] %s is installed, but you need to restart pd.",
                 pipPackage);
        PyErr_SetString(PyExc_TypeError,
                        "[Python] Installed, but you need to restart pd.");
        return NULL;
    }
    return NULL;
}

// =================================
// ========= MODULE INIT ===========
// =================================

#if PYTHON_REQUIRED_VERSION(3, 12)

static int _pd_create(PyObject *m) {
    (void)m;
    return 0;
}

static int _pd_modexec(PyObject *m) {
    (void)m;
    return 0;
}

static PyModuleDef_Slot _memoryboard_slots[] = {
    {Py_mod_create, _pd_create},
    {Py_mod_exec, _pd_modexec},
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
    {0, NULL}};

#endif

// =================================
static PyObject *pdmoduleError;

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
    {"logpost", Py4pdMod_PdLogPost, METH_VARARGS,
     "Print informations in PureData Console"},
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
    {"get_patch_dir", Py4pdMod_GetPatchHome, METH_VARARGS,
     "Get PureData Patch Path Folder"},
    {"get_home_folder", Py4pdMod_GetPatchHome, METH_VARARGS,
     "Get PureData Patch Path Folder"},
    {"get_py4pd_dir", Py4pdMod_GetObjFolder, METH_VARARGS,
     "Get PureData Py4PD Folder"},
    {"get_temp_dir", Py4pdMod_GetObjTmpFolder, METH_VARARGS,
     "Get PureData Temp Folder"},

    // User
    {"get_key", Py4pdMod_PdKey, METH_VARARGS, "Get Object User Parameters"},

    // pd
    {"get_sample_rate", Py4pdMod_PdSampleRate, METH_NOARGS,
     "Get PureData SampleRate"},
    {"get_vec_size", Py4pdMod_PdVecSize, METH_NOARGS,
     "Get PureData Vector Size"},
    {"get_num_channels", Py4pdMod_ObjNChannels, METH_NOARGS,
     "Return the amount of channels in the object"},
    {"pd_has_gui", Py4pdMod_PdHasGui, METH_NOARGS,
     "Return True of False if pd has or no gui"},

    {"get_patch_zoom", Py4pdMod_PdZoom, METH_NOARGS, "Get Patch zoom"},
    {"get_outlet_count", Py4pdMod_PdGetOutCount, METH_NOARGS,
     "Get the Number of Outlets of one object."},
    {"get_object_args", Py4pdMod_GetObjArgs, METH_NOARGS,
     "Returns list with all the args."},

    // library methods
    {"add_object", (PyCFunction)Py4pdLib_AddObj, METH_VARARGS | METH_KEYWORDS,
     "It adds python functions as objects"},

    // pip install
    {"pip_install", Py4pdMod_PipInstall, METH_VARARGS,
     "It installs a pip package"},

    // Others
    {"get_obj_pointer", Py4pdMod_GetObjPointer, METH_NOARGS,
     "Get PureData Object Pointer"},
    {"get_str_pointer", Py4pdMod_GetObjPointer, METH_NOARGS,
     "Get PureData Object Pointer"},
    {"set_obj_var", Py4pdMod_SetGlobalVar, METH_VARARGS,
     "It sets a global variable for the Object, it is not clear after the "
     "execution of the function"},

    // Loops
    {"get_obj_var", (PyCFunction)Py4pdMod_GetGlobalVar,
     METH_VARARGS | METH_KEYWORDS,
     "It gets a global variable for the Object, it is not clear after the "
     "execution of the function"},
    {"clear_obj_var", (PyCFunction)Py4pdMod_ClearGlobalVar, METH_VARARGS,
     "It clear the Dictionary of global variables"},
    {"accum_obj_var", Py4pdMod_AccumGlobalVar, METH_VARARGS,
     "It adds the values in the end of the list"},

    // player
    {"add_to_player", (PyCFunction)Py4pdMod_AddThingToPlay,
     METH_VARARGS | METH_KEYWORDS, "It adds a thing to the player"},
    {"clear_player", Py4pdMod_ClearPlayer, METH_NOARGS,
     "Remove all Python Objects of the player."},

    // Internal
    {"_recursive", Py4pdMod_PdRecursiveCall, METH_VARARGS,
     "It calls a function recursively"},

    {NULL, NULL, 0, NULL} //
};

// =================================
static struct PyModuleDef pdmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pd", /* name of module */
    .m_doc = "pd module provide function to interact with PureData, see the "
             "docs in www.charlesneimog.com/py4pd",
    .m_size = 0, /* size of per-interpreter state of the module, or -1 if
                    the module keeps state in global variables. */
    .m_methods = PdMethods, // Methods of the module

#if PYTHON_REQUIRED_VERSION(3, 12)
    .m_slots = _memoryboard_slots,
#endif

    .m_traverse = NULL, /* m_traverse, that is the traverse function for GC */
    .m_clear = NULL,    /* m_free, that is the free function for GC */
};

// =================================
const char *createUniqueConstString(const char *prefix, const void *pointer) {
    char buffer[32]; // Adjust the buffer size as per your requirement
    snprintf(buffer, sizeof(buffer), "%s_%p", prefix, pointer);
    char *uniqueConstString = (char *)malloc(strlen(buffer) + 1);
    if (uniqueConstString != NULL) {
        strcpy(uniqueConstString, buffer);
    }
    return uniqueConstString;
}

// =================================
PyMODINIT_FUNC PyInit_pd() {
    PyObject *py4pdmodule;

    py4pdmodule = PyModule_Create(&pdmodule);

    if (py4pdmodule == NULL) {
        return NULL;
    }

    // create one string from py4pdmodule
    const char *OUT_string = createUniqueConstString("py4pdOut", py4pdmodule);
    const char *CLEAR_string =
        createUniqueConstString("py4pdClear", py4pdmodule);

    PyObject *puredata_samplerate, *puredata_vecsize, *visObject, *normalObject,
        *audioINObject, *audioOUTObject, *pdAudio, *Py4pd_OutLoopString,
        *Py4pd_ClearLoopString;

    puredata_samplerate = PyLong_FromLong(sys_getsr());
    puredata_vecsize = PyLong_FromLong(sys_getblksize());

    normalObject = PyLong_FromLong(PY4PD_NORMALOBJ);
    visObject = PyLong_FromLong(PY4PD_VISOBJ);
    audioINObject = PyLong_FromLong(PY4PD_AUDIOINOBJ);
    audioOUTObject = PyLong_FromLong(PY4PD_AUDIOOUTOBJ);
    pdAudio = PyLong_FromLong(PY4PD_AUDIOOBJ);

    Py4pd_OutLoopString = PyUnicode_FromString(OUT_string);
    Py4pd_ClearLoopString = PyUnicode_FromString(CLEAR_string);

    PyModule_AddObject(py4pdmodule, "SAMPLERATE", puredata_samplerate);
    PyModule_AddObject(py4pdmodule, "VECSIZE", puredata_vecsize);

    PyModule_AddObject(py4pdmodule, "NORMAL", normalObject);
    PyModule_AddObject(py4pdmodule, "VIS", visObject);
    PyModule_AddObject(py4pdmodule, "AUDIOIN", audioINObject);
    PyModule_AddObject(py4pdmodule, "AUDIOOUT", audioOUTObject);
    PyModule_AddObject(py4pdmodule, "AUDIO", pdAudio);

    PyModule_AddObject(py4pdmodule, "OUTLOOP", Py4pd_OutLoopString);
    PyModule_AddObject(py4pdmodule, "CLEARLOOP", Py4pd_ClearLoopString);

    pdmoduleError = PyErr_NewException("spam.error", NULL, NULL);

    Py_XINCREF(pdmoduleError);
    if (PyModule_AddObject(py4pdmodule, "moduleerror", pdmoduleError) < 0) {
        Py_XDECREF(pdmoduleError);
        Py_CLEAR(pdmoduleError);
        Py_DECREF(py4pdmodule);
        return NULL;
    }

    if (PyType_Ready(&Py4pdNewObj_Type) < 0)
        return NULL;

    Py_INCREF(&Py4pdNewObj_Type);
    PyModule_AddObject(py4pdmodule, "NewObject", (PyObject *)&Py4pdNewObj_Type);

    return py4pdmodule;
}

// ============================
