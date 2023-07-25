#include "py4pd.h"
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#include <numpy/arrayobject.h>


// =================================
static pdcollectHash* CreatePdcollectHash(int size) { // TODO: make thing to free table when object is deleted?
    pdcollectHash* hash_table = (pdcollectHash*)malloc(sizeof(pdcollectHash));
    hash_table->size = size;
    hash_table->count = 0;
    hash_table->items = (pdcollectItem**)calloc(size, sizeof(pdcollectItem*));
    return hash_table;
}

// =================================
static unsigned int HashFunction(pdcollectHash* hash_table, char* key) {
    unsigned long hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % hash_table->size;
}

// =================================
static void InsertItem(pdcollectHash* hash_table, char* key, PyObject* obj) {
    unsigned int index = HashFunction(hash_table, key);
    pdcollectItem* item = hash_table->items[index];
    hash_table->count++; // TODO: Make resizeable table. For now we have 8 items.
    if (item == NULL &&  hash_table->count <= hash_table->size) {
        item = (pdcollectItem*)malloc(sizeof(pdcollectItem));
        item->key = strdup(key);
        item->pItem = obj;
        hash_table->items[index] = item;
        return;
    }
    else if (hash_table->count > hash_table->size) {
        PyErr_SetString(PyExc_MemoryError, "[Python] pd.setglobalvar: memory error");
        return;
    }
    else if (item != NULL) {
        Py4pdUtils_DECREF(item->pItem); // Decrement the reference count of the object.
        item->pItem = obj;
        return;
    }
}

// =================================
static void AccumItem(pdcollectHash* hash_table, char* key, PyObject* obj) {
    unsigned int index = HashFunction(hash_table, key);
    pdcollectItem* item = hash_table->items[index];
    if (item == NULL &&  hash_table->count <= hash_table->size) {
        item = (pdcollectItem*)malloc(sizeof(pdcollectItem));
        item->key = strdup(key);
        item->pList = PyList_New(0);
        hash_table->items[index] = item;
    }
    else if (hash_table->count > hash_table->size) {
        PyErr_SetString(PyExc_MemoryError, "[Python] pd.setglobalvar: memory error");
        return;
    }
    if (item->wasCleaned) 
        item->wasCleaned = 0;
    PyList_Append(item->pList, obj);
    return;
}


// =================================
static void ClearItem(pdcollectHash* hash_table, char* key) {
    unsigned int index = HashFunction(hash_table, key);
    pdcollectItem* item = hash_table->items[index];
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
static void ClearList(pdcollectHash* hash_table, char* key) {
    PY4PD_FUNC_CALL();
    unsigned int index = HashFunction(hash_table, key);
    pdcollectItem* item = hash_table->items[index];
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
// =================================
static pdcollectItem* GetObjArr(pdcollectHash* hash_table, char* key) {
    unsigned int index = HashFunction(hash_table, key);
    pdcollectItem* item = hash_table->items[index];
    if (item == NULL) 
        return NULL;
    return item;
}

// =================================
// =================================
// =================================
static PyObject *Py4pdMod_SetGlobalVar(PyObject *self, PyObject *args){
    PY4PD_FUNC_CALL();
    (void)self;
    t_py *py4pd = Py4pdUtils_GetObject();
    if (py4pd == NULL){
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
    snprintf(key, strlen(varName) + 40, "%s_%p", varName, py4pd);
    if (py4pd->pdcollect == NULL){
        py4pd->pdcollect = CreatePdcollectHash(8);
    }
    InsertItem(py4pd->pdcollect, key, pValueScript);
    free(key);
    Py_RETURN_TRUE;
}

// =================================
static PyObject *Py4pdMod_GetGlobalVar(PyObject *self, PyObject *args, PyObject *keywords){
    PY4PD_FUNC_CALL();



    (void)self;
    t_py *py4pd = Py4pdUtils_GetObject();
    if (py4pd == NULL){
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
    snprintf(key, strlen(varName) + 40, "%s_%p", varName, py4pd);
    if (py4pd->pdcollect == NULL){
        py4pd->pdcollect = CreatePdcollectHash(8); // TODO: add way to resize
    }

    if (keywords != NULL) {
        PyObject *pValueInit = PyDict_GetItemString(keywords, "initial_value");
        if (pValueInit != NULL) 
            InsertItem(py4pd->pdcollect, key, pValueInit);
    }

    pdcollectItem* item = GetObjArr(py4pd->pdcollect, key); 
    if (item == NULL) {
        free(key);
        Py_RETURN_NONE;
    }
    free(key);
    if (item->aCumulative){
        return item->pList;
    }
    else{
        if (item->pList == Py_None)
            Py_RETURN_NONE;
        return item->pItem;
    }
}

// =================================
static PyObject *Py4pdMod_AccumGlobalVar(PyObject *self, PyObject *args){
    PY4PD_FUNC_CALL();
    (void)self;

    t_py *py4pd = Py4pdUtils_GetObject();
    if (py4pd == NULL){
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
    snprintf(key, strlen(varName) + 40, "%s_%p", varName, py4pd);
    if (py4pd->pdcollect == NULL){
        py4pd->pdcollect = CreatePdcollectHash(8);
    }
    AccumItem(py4pd->pdcollect, key, pValueScript);
    py4pd->pdcollect->items[HashFunction(py4pd->pdcollect, key)]->aCumulative = 1;
    free(key);
    PY4PD_FUNC_DONE();
    Py_RETURN_TRUE;

}

// =================================
static PyObject *Py4pdMod_ClearGlobalVar(PyObject *self, PyObject *args) {
    PY4PD_FUNC_CALL();
    (void)self;

    t_py *py4pd = Py4pdUtils_GetObject();
    if (py4pd == NULL){
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
    snprintf(key, strlen(varName) + 40, "%s_%p", varName, py4pd);
    if (py4pd->pdcollect == NULL){
        free(key);
        PY4PD_FUNC_DONE();
        Py_RETURN_TRUE;
    }
    pdcollectItem* objArr = GetObjArr(py4pd->pdcollect, key);
    if (objArr != NULL){
        if (objArr->wasCleaned){
            free(key);
            PY4PD_FUNC_DONE();
            Py_RETURN_TRUE;
        }
    }
    else{
        free(key);
        PY4PD_FUNC_DONE();
        Py_RETURN_TRUE;
    }

    if (objArr->aCumulative) 
        ClearList(py4pd->pdcollect, key);
    else
        ClearItem(py4pd->pdcollect, key);

    free(key);
    PY4PD_FUNC_DONE();
    Py_RETURN_TRUE;
}

// ======================================
static void Py4pdMod_RecursiveTick(t_py *x){
    t_py4pd_pValue *pdPyValue = (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
    pdPyValue->pValue = x->recursiveObject;
    pdPyValue->objectsUsing = 0;
    Py4pdUtils_ConvertToPd(x, pdPyValue, x->out1);
    Py_DECREF(pdPyValue->pValue); // delete thing
    free(pdPyValue);
    Py_LeaveRecursiveCall();
    // clock_unset(x->recursiveClock);
}


// ======================================
static PyObject *Py4pdMod_PdRecursiveCall(PyObject *self, PyObject *args){
    PY4PD_FUNC_CALL();
    (void)self;

    PyObject* pValue;
    if (!PyArg_ParseTuple(args, "O", &pValue)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.setglobalvar: wrong arguments");
        return NULL;
    }

    t_py *py4pd = Py4pdUtils_GetObject();
    py4pd->recursiveCalls++;

    Py_EnterRecursiveCall("[py4pd] Exceeded maximum recursion depth");
    if (py4pd->recursiveCalls == py4pd->stackLimit){ // seems to be the limit in PureData, 
        py4pd->recursiveObject = pValue;
        if (py4pd->recursiveClock == NULL)
            py4pd->recursiveClock = clock_new(py4pd, (t_method)Py4pdMod_RecursiveTick);
        Py_INCREF(pValue); // avoid thing to be deleted
        py4pd->recursiveCalls = 0;
        clock_delay(py4pd->recursiveClock, 0);
        Py_RETURN_TRUE;
    }
    
    if (py4pd == NULL){
        PyErr_SetString(PyExc_RuntimeError, "[pd._recursive] pd.recursive: py4pd is NULL");
        return NULL;
    }

    t_py4pd_pValue *pdPyValue = (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
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
static PyObject *Py4pdMod_PdGetOutCount(PyObject *self, PyObject *args){
    (void)self;
    (void)args;
    t_py *py4pd = Py4pdUtils_GetObject();
    if (py4pd == NULL){
        PyErr_SetString(PyExc_RuntimeError, "[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }
    return PyLong_FromLong(py4pd->outAUX->u_outletNumber);
}

// =================================
static PyObject *Py4pdMod_PdOut(PyObject *self, PyObject *args, PyObject *keywords){
    PY4PD_FUNC_CALL();
    (void)self;

    t_py *py4pd = Py4pdUtils_GetObject();
    if (py4pd == NULL){
        PyErr_SetString(PyExc_RuntimeError, "[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }

    PyObject* pValue;

    if (!PyArg_ParseTuple(args, "O", &pValue)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.out: wrong arguments");
        return NULL;
    }

    t_py4pd_pValue *pdPyValue = (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
    pdPyValue->pValue = pValue;
    pdPyValue->objectsUsing = 0;

    if (keywords != NULL && py4pd->outAUX != NULL){
        PyObject *outletNumber = PyDict_GetItemString(keywords, "out_n"); // it gets the data type output
        if (outletNumber == NULL){
            Py4pdUtils_ConvertToPd(py4pd, pdPyValue, py4pd->out1);
            free(pdPyValue);
            Py_RETURN_TRUE;
        }

        if (!PyLong_Check(outletNumber)){
            PyErr_SetString(PyExc_TypeError, "[Python] pd.out: out_n must be an integer.");
            Py_DECREF(pValue);
            return NULL;
        }
        int outletNumberInt = PyLong_AsLong(outletNumber);
        if (outletNumberInt == 0){
            Py4pdUtils_ConvertToPd(py4pd, pdPyValue, py4pd->out1);
            free(pdPyValue);
            Py_RETURN_TRUE;
        }
        else{
            outletNumberInt--;
            if ((py4pd->outAUX->u_outletNumber > 0) && (outletNumberInt < py4pd->outAUX->u_outletNumber)){
                Py4pdUtils_ConvertToPd(py4pd, pdPyValue, py4pd->outAUX[outletNumberInt].u_outlet);
                free(pdPyValue);
                Py_RETURN_TRUE;
            }
            else{
                PyErr_SetString(PyExc_TypeError, "[Python] pd.out: Please check the number of outlets."); 
                return NULL;
            }
        }
    }
    else{
        Py4pdUtils_ConvertToPd(py4pd, pdPyValue, py4pd->out1);
        free(pdPyValue);
    }
    Py_RETURN_TRUE;
}
// =================================
static PyObject *Py4pdMod_PdPrint(PyObject *self, PyObject *args, PyObject *keywords) {
    (void)self;
    int printPrefix = 1;
    int objPrefix = 1;

    t_py *py4pd = Py4pdUtils_GetObject();
    if (py4pd == NULL){
        PyErr_SetString(PyExc_RuntimeError, "[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }

    if (keywords == NULL) {
        printPrefix = 1;
    } 
    else {
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
                pd_error(NULL, "[Python] pd.print: show_prefix argument must be True or False.");
                printPrefix = 1;
            }
        } 
        else 
            printPrefix = 1;
        if (objPrefix == -1) 
            post("[Python] pd.print: error in obj_prefix argument.");
        else if (objPrefix == 1) {
            PyObject *resize_value = PyDict_GetItemString(keywords, "obj_prefix");
            if (resize_value == Py_True) 
                objPrefix = 1;
            else if (resize_value == Py_False)
                objPrefix = 0;
            else {
                post("[Python] pd.print: obj_prefix argument must be True or False.");
                objPrefix = 0;
            }
        } 
        else 
            objPrefix = 0;
    }

    PyObject* obj;
    if (PyArg_ParseTuple(args, "O", &obj)) {
        PyObject* str = PyObject_Str(obj);
        if (str == NULL) {
            Py_DECREF(str);
            PyErr_SetString(PyExc_TypeError, "[Python] pd.print failed to convert object to string.");
            return NULL;
        }
        const char* str_value = PyUnicode_AsUTF8(str);
        if (str_value == NULL) {
            PyErr_SetString(PyExc_TypeError, "[Python] pd.print failed to convert string object to UTF-8.");
            Py_DECREF(str);
            return NULL;
        }
        if (printPrefix == 1) { //
            if (py4pd->objectName == NULL){
                post("[Python]: %s", str_value);
            }
            else{
                post("[%s]: %s", py4pd->objectName->s_name, str_value);
            }
            Py_RETURN_TRUE;
        } 
        else {
            post("%s", str_value);
        }
        Py_DECREF(str);
    } 
    else {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.print works with strings, numbers, and any other valid Python object.");
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
    }
    else {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.logpost works with strings and numbers.");
        return NULL;
    }
}


// =================================
static PyObject *Py4pdMod_PdError(PyObject *self, PyObject *args) {
    (void)self;

    char *string;

    t_py *py4pd = Py4pdUtils_GetObject();

    if (PyArg_ParseTuple(args, "s", &string)) {
        if (py4pd == NULL){
            pd_error(NULL, "%s", string);
            PyErr_Clear();
            return PyLong_FromLong(0);
        }

        if (py4pd->pyObject == 1){
            pd_error(py4pd, "[%s]: %s", py4pd->objectName->s_name, string);
        }
        else{
            if (py4pd->function_name == NULL){
                pd_error(py4pd, "%s", string);
            }
            else{
                pd_error(py4pd, "[%s]: %s", py4pd->function_name->s_name, string);
            }
        }
        PyErr_Clear();
    } 
    else {
        PyErr_SetString(
            PyExc_TypeError,
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
        } 
        else {
            post("[Python] pd.send not found object [r %s] in pd patch",
                 receiver);
        }
    } 
    else if (PyArg_ParseTuple(args, "sf", &receiver, &floatNumber)) {
        t_symbol *symbol = gensym(receiver);
        if (symbol->s_thing) {
            pd_float(symbol->s_thing, floatNumber);
        } 
        else {
            post("[Python] pd.send not found object [r %s] in pd patch",
                 receiver);
        }
    } else if (PyArg_ParseTuple(args, "si", &receiver, &intNumber)) {
        t_symbol *symbol = gensym(receiver);
        if (symbol->s_thing) {
            pd_float(symbol->s_thing, intNumber);
        } 
        else {
            post("[Python] pd.send not found object [r %s] in pd patch",
                 receiver);
        }
    } 
    else if (PyArg_ParseTuple(args, "sO", &receiver, &listargs)) {
        if (PyDict_Check(listargs)) {
            char error_message[100];
            sprintf(error_message,
                    "[Python] pd.send received a type 'dict', it must be a list, "
                    "string, int, or float.");
            PyErr_SetString(PyExc_TypeError,
                           error_message);  
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
            } 
            else if (PyFloat_Check(pValue_i)) {  
                                                  
                double result = PyFloat_AsDouble(pValue_i);
                float result_float = (float)result;
                list_array[i].a_type = A_FLOAT;
                list_array[i].a_w.w_float = result_float;
            } 
            else if (PyUnicode_Check(pValue_i)) { 
                const char *result = PyUnicode_AsUTF8(pValue_i);
                list_array[i].a_type = A_SYMBOL;
                list_array[i].a_w.w_symbol = gensym(result);
            } 
            else if (Py_IsNone(pValue_i)) {
                // post("None");
            }
            else {
                char error_message[100];
                sprintf(error_message,
                        "[Python] received a type '%s' in index %d of the "
                        "list, it must be a string, int, or float.",
                        pValue_i->ob_type->tp_name, i);
                PyErr_SetString(PyExc_TypeError,
                                error_message);  
                Py_DECREF(pValue_i);
                Py_DECREF(args);
                free(list_array);
                return NULL;
            }
            // Py_DECREF(pValue_i);
        }
        if (gensym(receiver)->s_thing) {
            pd_list(gensym(receiver)->s_thing, &s_list, list_size, list_array);
        } 
        else {
            pd_error(NULL, "[Python] object [r %s] not found", receiver);
        }
    } 
    else {
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
static PyObject *Py4pdMod_PdTabWrite(PyObject *self, PyObject *args, PyObject *keywords) {
    (void)self;
    int resize = 0;
    int vecsize;
    t_garray *pdarray;
    t_word *vec;
    char *string;
    PyObject *PYarray;

    if (keywords == NULL) {
        resize = 0;
        PyErr_Clear();
    } 
    else {

        resize = PyDict_Contains(keywords, PyUnicode_FromString("resize"));
        if (resize == -1) {
            post("error");
        } 
        else if (resize == 1) {
            PyObject *resize_value = PyDict_GetItemString(keywords, "resize");
            if (resize_value == Py_True) {
                resize = 1;
            } 
            else if (resize_value == Py_False) {
                resize = 0;
            } 
            else {
                resize = 0;
            }
        } 
        else {
            resize = 0;
        }
    }

    // ================================
    t_py *py4pd = Py4pdUtils_GetObject();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }

    // ================================

    if (PyArg_ParseTuple(args, "sO", &string, &PYarray)) {
        t_symbol *pd_symbol = gensym(string);
        if (!(pdarray = (t_garray *)pd_findbyclass(pd_symbol, garray_class)))
            pd_error(py4pd, "[Python] Array %s not found.", string);
        else if (!garray_getfloatwords(pdarray, &vecsize, &vec))
            pd_error(py4pd, "[Python] Bad template for tabwrite '%s'.",
                     string);
        else { // TODO: Add support to numpy arrays
            int i;
            if (resize == 1) {
                garray_resize_long(pdarray, PyList_Size(PYarray));
                vecsize = PyList_Size(PYarray);
                garray_getfloatwords(pdarray, &vecsize, &vec);
            }
            for (i = 0; i < vecsize; i++) {
                double result = PyFloat_AsDouble(PyList_GetItem(PYarray, i));
                float result_float = (float)result;
                vec[i].w_float = result_float;
            }
            garray_redraw(pdarray);
            PyErr_Clear();
        }
    }
    Py_RETURN_TRUE;
}

// =================================
static PyObject *Py4pdMod_PdTabRead(PyObject *self, PyObject *args, PyObject *keywords) {
    (void)self;
    int vecsize;
    t_garray *pdarray;
    t_word *vec;
    char *string;
    int numpy;

    // ================================
    t_py *py4pd = Py4pdUtils_GetObject();
    // ================================

    if (keywords == NULL) {
        numpy = 0;
        PyErr_Clear();
    } 
    else{
        numpy = PyDict_Contains(keywords, PyUnicode_FromString("numpy"));
        if (numpy == -1) {
            pd_error(NULL, "[Python] Check the keyword arguments.");
            return NULL;
        } 
        else if (numpy == 1) {
            PyObject *numpy_value = PyDict_GetItemString(keywords, "numpy");
            if (numpy_value == Py_True) {
                numpy = 1;
                int numpyArrayImported = _import_array();
                if (numpyArrayImported == 0) {
                    py4pd->numpyImported = 1;
                }
                else{
                    py4pd->numpyImported = 0;
                    pd_error(py4pd, "[py4pd] Not possible to import numpy array");
                    return NULL;
                }
            } 
            else if (numpy_value == Py_False) {
                numpy = 0;
            } 
            else {
                numpy = 0;
            }
        } 
        else {
            numpy = 0;
        }
    }

    if (PyArg_ParseTuple(args, "s", &string)) {
        t_symbol *pd_symbol = gensym(string);
        if (!(pdarray = (t_garray *)pd_findbyclass(pd_symbol, garray_class))) {
            pd_error(py4pd, "[Python] Array %s not found.", string);
            PyErr_SetString(PyExc_TypeError,
                            "[Python] pd.tabread: array not found");
        } 
        else {
            int i;
            if (numpy == 0) {
                garray_getfloatwords(pdarray, &vecsize, &vec);
                PyObject *list = PyList_New(vecsize);
                for (i = 0; i < vecsize; i++) {
                    PyList_SetItem(list, i, PyFloat_FromDouble(vec[i].w_float));
                }
                PyErr_Clear();
                return list;
            }
            else if (numpy == 1) {
                garray_getfloatwords(pdarray, &vecsize, &vec);
                const npy_intp dims = vecsize;
                // send double float array to numpy
                PyObject *array = PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT, vec);
                PyErr_Clear();
                return array;

            }
            else {
                pd_error(py4pd, "[Python] Check the keyword arguments.");
                return NULL;
            }
        }
    } 
    else {
        PyErr_SetString(PyExc_TypeError,
                        "[Python] pd.tabread: wrong arguments");
        return NULL;
    }
    return NULL;
}

// =================================
static PyObject *Py4pdMod_GetPatchHome(PyObject *self, PyObject *args) {
    (void)self;

    t_py *py4pd = Py4pdUtils_GetObject();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
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

    t_py *py4pd = Py4pdUtils_GetObject();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
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
        PyErr_SetString(PyExc_TypeError, "[Python] pd.samplerate: no argument expected");
        return NULL;
    }
    t_py *py4pd = Py4pdUtils_GetObject();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }
    Py4pdUtils_CreateTempFolder(py4pd);
    return PyUnicode_FromString(py4pd->tempPath->s_name);
}

// =================================
static PyObject *Py4pdMod_ShowImage(PyObject *self, PyObject *args) {
    (void)self;
    char *string;
    t_py *py4pd = Py4pdUtils_GetObject();
    if (py4pd == NULL){
        PyErr_SetString(PyExc_TypeError,
                            "[Python] py4pd capsule not found");
        return NULL;
    }
    if (PyArg_ParseTuple(args, "s", &string)) {
        t_symbol *filename = gensym(string);
        if (py4pd->x_def_img) {
            py4pd->x_def_img = 0;
        }
        if(access(filename->s_name, F_OK) == -1) {
            pd_error(py4pd, "[Python] File %s not found.", filename->s_name);

            // reset image to default
            py4pd->x_def_img = 1;
            // py4pd->x_width = 250;
            // py4pd->x_height = 250;
            Py4pdPic_Erase(py4pd, py4pd->glist);
            sys_vgui(".x%lx.c itemconfigure %lx_picture -image PY4PD_IMAGE_{%p}\n", py4pd->canvas, py4pd, py4pd);
            Py4pdPic_Draw(py4pd, py4pd->glist, 1);
            Py_RETURN_NONE;
        }

        FILE *file;
        char *ext = strrchr(filename->s_name, '.');
        if (strcmp(ext, ".ppm") == 0) {
            char magic_number[3];
            int width, height, max_color_value;
            file = fopen(filename->s_name, "r");
            fscanf(file, "%s\n%d %d\n%d\n", magic_number, &width, &height, &max_color_value);
            py4pd->x_width = width;
            py4pd->x_height = height;
            fclose(file);
        }
        else if (strcmp(ext, ".gif") == 0){
            file = fopen(filename->s_name, "rb");
            fseek(file, 6, SEEK_SET);
            fread(&py4pd->x_width, 2, 1, file);
            fread(&py4pd->x_height, 2, 1, file);
            fclose(file);
        }
        else if (strcmp(ext, ".png") == 0) {
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

        else{
            pd_error(py4pd, "[Python] pd.showimage: file format not supported");
            PyErr_SetString(PyExc_TypeError, "[Python] pd.showimage: file format not supported");
            return NULL;
        }

        if (glist_isvisible(py4pd->glist) && gobj_shouldvis((t_gobj *)py4pd, py4pd->glist)) {
            const char *file_name_open = Py4pdPic_Filepath(py4pd, filename->s_name);
            if(access(file_name_open, F_OK) == -1) {
                pd_error(py4pd, "[Python] pd.showimage: file not found");
                PyErr_SetString(PyExc_TypeError, "[Python] pd.showimage: file not found");
                return NULL;
            }
            if (file_name_open) {
                py4pd->x_filename = filename;
                py4pd->x_fullname = gensym(file_name_open);

                if (py4pd->x_def_img) {
                    py4pd->x_def_img = 0;
                }

                if (glist_isvisible(py4pd->glist) && gobj_shouldvis((t_gobj *)py4pd, py4pd->glist)) {
                    Py4pdPic_Erase(py4pd, py4pd->glist);
                    sys_vgui(
                        "if {[info exists %lx_picname] == 0} {image create "
                        "photo %lx_picname -file \"%s\"\n set %lx_picname "
                        "1\n}\n",
                        py4pd->x_fullname, py4pd->x_fullname, file_name_open,
                        py4pd->x_fullname);
                    Py4pdPic_Draw(py4pd, py4pd->glist, 1);
                }
            } 
            else {
                pd_error(py4pd, "[Python]: Error displaying image, file not found");
                PyErr_Clear();
                Py_RETURN_NONE;
            }
        } 
        else {
            pd_error(py4pd, "[Python]: Error displaying image");
            PyErr_Clear();
            Py_RETURN_NONE;
        }
    } 
    else {
        pd_error(py4pd, "[Python] pd.showimage received wrong arguments");
        PyErr_Clear();
        Py_RETURN_NONE;
    }
    PyErr_Clear();
    Py_RETURN_NONE;
}

// =================================
// ========== AUDIO CONFIG =========
// =================================

static PyObject* Py4pdMod_PdSampleRate(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;
    float sr = sys_getsr(); // call PureData's sys_getsr function to get the current sample rate
    return PyFloat_FromDouble(sr); // return the sample rate as a Python float object
}

// =================================
static PyObject *Py4pdMod_PdVecSize(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    t_py *py4pd = Py4pdUtils_GetObject();
    t_sample vector;
    vector = py4pd->vectorSize; // this is the patch vector size
    // TODO: Should I add someway to get the puredata's vector size? with sys_getblksize()?

    return PyLong_FromLong(vector);
}

// =================================
static PyObject *Py4pdMod_PdZoom(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    t_py *py4pd = Py4pdUtils_GetObject();
    int zoom;
    if (py4pd->canvas != NULL) {
        zoom = (int)py4pd->canvas->gl_zoom;
    }
    else {
        pd_error(NULL, "[Python] pd.patchzoom: canvas not found");
        zoom = 1;
    }
    return PyLong_FromLong(zoom);
}

// =================================
// ========== Utilities ============
// =================================

static PyObject *Py4pdMod_PdKey(PyObject *self, PyObject *args) {
    //get values from Dict salved in x->param
    (void)self;
    char *key;
    if (!PyArg_ParseTuple(args, "s", &key)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.key: no argument expected");
        return NULL;
    }
    
    t_py *py4pd = Py4pdUtils_GetObject();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
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
static PyObject *Py4pdMod_GetObjPointer(PyObject *self, PyObject *args){
    (void)self;
    (void)args;
    
    t_py *py4pd = Py4pdUtils_GetObject();
    if (py4pd == NULL){
        post("[Python] py4pd capsule not found. The module pd must be used inside py4pd object or functions.");
        return NULL;
    }
    
    return PyUnicode_FromFormat("%p", py4pd);
}

// =================================
// ============ PLAYER =============
// =================================
static PyObject *Py4pdMod_AddThingToPlay(PyObject *self, PyObject *args, PyObject *keywords){
    (void)self;
    (void)keywords;

    int onset;
    PyObject *thingToPlay; 
    t_py *py4pd = Py4pdUtils_GetObject();

    if (!PyArg_ParseTuple(args, "iO", &onset, &thingToPlay)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.add2play: wrong arguments");
        return NULL;
    }
    Py4pdLib_PlayerInsertThing(py4pd, onset, Py_BuildValue("O", thingToPlay)); 
    Py_RETURN_NONE;
}

// =================================
static PyObject *Py4pdMod_ClearPlayer(PyObject *self, PyObject *args){
    (void)self;
    (void)args;

    t_py *py4pd = Py4pdUtils_GetObject();
    Py4pdLib_Clear(py4pd);
    Py_RETURN_NONE;
}

// =================================
// ============= PIP ===============
// =================================
static PyObject *Py4pdMod_PipInstall(PyObject *self, PyObject *args){
    (void)self;
    char *pipPackage;
    char *localORglobal;
    if (!PyArg_ParseTuple(args, "ss", &localORglobal, &pipPackage)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.pipInstall: wrong arguments");
        return NULL;
    }

    PyObject *py4pdModule = PyImport_ImportModule("py4pd");
    if (py4pdModule == NULL) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.pipInstall: py4pd module not found");
        return NULL;
    }
    PyObject *pipInstallFunction = PyObject_GetAttrString(py4pdModule, "pipinstall");
    if (pipInstallFunction == NULL) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.pipInstall: pipinstall function not found");
        return NULL;
    }


    // the function is executed using pipinstall([localORglobal, pipPackage])
    PyObject *argsList = PyList_New(2);
    PyList_SetItem(argsList, 0, Py_BuildValue("s", localORglobal));
    PyList_SetItem(argsList, 1, Py_BuildValue("s", pipPackage));
    PyObject *argTuple = Py_BuildValue("(O)", argsList);
    PyObject *pipInstallResult = PyObject_CallObject(pipInstallFunction, argTuple);

    if (pipInstallResult == NULL) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.pipInstall: pipinstall function failed");
        return NULL;
    }
    Py_DECREF(argTuple);
    Py_DECREF(pipInstallResult);
    Py_DECREF(pipInstallFunction);
    Py_DECREF(py4pdModule);
    sys_vgui("tk_messageBox -icon warning -type ok -title \"%s installed!\" -message \"%s installed! \nYou need to restart PureData!\"\n", pipPackage, pipPackage);
    Py_RETURN_TRUE;
}

// =================================
static PyObject *pdmoduleError;

// =================================
PyMethodDef PdMethods[] = {

    // PureData inside Python
    {"out", (PyCFunction)Py4pdMod_PdOut, METH_VARARGS | METH_KEYWORDS, "Output in out0 from PureData"},
    {"send", Py4pdMod_PdSend, METH_VARARGS, "Send message to PureData, it can be received with the object [receive]"},
    {"print", (PyCFunction)Py4pdMod_PdPrint, METH_VARARGS | METH_KEYWORDS,  "Print informations in PureData Console"},
    {"logpost", Py4pdMod_PdLogPost, METH_VARARGS, "Print informations in PureData Console"},
    {"error", Py4pdMod_PdError, METH_VARARGS, "Print informations in error format (red) in PureData Console"},
    {"tabwrite", (PyCFunction)Py4pdMod_PdTabWrite, METH_VARARGS | METH_KEYWORDS, "Write data to PureData tables/arrays"},
    {"tabread", (PyCFunction)Py4pdMod_PdTabRead, METH_VARARGS | METH_KEYWORDS, "Read data from PureData tables/arrays"},
    
    // Pic
    {"show", Py4pdMod_ShowImage, METH_VARARGS, "Show image in PureData, it must be .gif, .bmp, .ppm"},

    // Files
    {"home", Py4pdMod_GetPatchHome, METH_VARARGS, "Get PureData Patch Path Folder"},
    {"homefolder", Py4pdMod_GetPatchHome, METH_VARARGS, "Get PureData Patch Path Folder"},
    {"py4pdfolder", Py4pdMod_GetObjFolder, METH_VARARGS, "Get PureData Py4PD Folder"},
    {"tempfolder", Py4pdMod_GetObjTmpFolder, METH_VARARGS, "Get PureData Temp Folder"},

    // User
    {"getkey", Py4pdMod_PdKey, METH_VARARGS, "Get Object User Parameters"},

    // pd
    {"samplerate", Py4pdMod_PdSampleRate, METH_NOARGS, "Get PureData SampleRate"},
    {"vecsize", Py4pdMod_PdVecSize, METH_NOARGS, "Get PureData Vector Size"},
    {"patchzoom", Py4pdMod_PdZoom, METH_NOARGS, "Get Patch zoom"},
    {"get_out_count", Py4pdMod_PdGetOutCount, METH_NOARGS, "Get the Number of Outlets of one object."},

    // library methods
    {"addobject", (PyCFunction)Py4pdLib_AddObj, METH_VARARGS | METH_KEYWORDS, "It adds python functions as objects"},

    // pip install
    {"pipinstall", Py4pdMod_PipInstall, METH_VARARGS, "It installs a pip package"},

    // Others
    {"getobjpointer", Py4pdMod_GetObjPointer, METH_NOARGS, "Get PureData Object Pointer"},
    {"getstrpointer", Py4pdMod_GetObjPointer, METH_NOARGS, "Get PureData Object Pointer"},
    {"setglobalvar", Py4pdMod_SetGlobalVar, METH_VARARGS, "It sets a global variable for the Object, it is not clear after the execution of the function"},

    // Loops
    {"getglobalvar", (PyCFunction)Py4pdMod_GetGlobalVar, METH_VARARGS | METH_KEYWORDS, "It gets a global variable for the Object, it is not clear after the execution of the function"},
    {"clearglobalvar", (PyCFunction)Py4pdMod_ClearGlobalVar, METH_VARARGS, "It clear the Dictionary of global variables"},
    {"accumglobalvar", Py4pdMod_AccumGlobalVar, METH_VARARGS, "It adds the values in the end of the list"},
    {"_recursive", Py4pdMod_PdRecursiveCall, METH_VARARGS, "It calls a function recursively"},


    // player
    {"add2player", (PyCFunction)Py4pdMod_AddThingToPlay, METH_VARARGS | METH_KEYWORDS, "It adds a thing to the player"},
    {"clearplayer", Py4pdMod_ClearPlayer, METH_NOARGS, "Get PureData Object Pointer"},

    // Internal
    // {"_iterate", Py4pdMod_PdIterate, METH_VARARGS, "Special method used to pd.iterate"},


    {NULL, NULL, 0, NULL}  //
};

// =================================
static struct PyModuleDef pdmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pd", /* name of module */
    .m_doc = "pd module provide function to interact with PureData, see the docs in www.charlesneimog.com/py4pd",
    .m_size = 0,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    .m_methods = PdMethods,  // Methods of the module
    .m_slots = NULL, /* m_slots, that is the slots for multi-phase initialization */
    .m_traverse = NULL, /* m_traverse, that is the traverse function for GC */
    .m_clear = NULL, /* m_free, that is the free function for GC */
};

// =================================
const char* createUniqueConstString(const char* prefix, const void* pointer) {
    char buffer[32]; // Adjust the buffer size as per your requirement
    snprintf(buffer, sizeof(buffer), "%s_%p", prefix, pointer);
    char* uniqueConstString = (char*)malloc(strlen(buffer) + 1);
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
    const char* OUT_string = createUniqueConstString("py4pdOut", py4pdmodule);
    const char* CLEAR_string = createUniqueConstString("py4pdClear", py4pdmodule);

    PyObject *puredata_samplerate, 
    *puredata_vecsize, *visObject, 
    *audioINObject, *audioOUTObject, 
    *pdAudio,
    *Py4pd_OutLoopString, *Py4pd_ClearLoopString;

    puredata_samplerate = PyLong_FromLong(sys_getsr());
    puredata_vecsize = PyLong_FromLong(sys_getblksize());
    
    visObject = PyUnicode_FromString("VIS");
    audioINObject = PyUnicode_FromString("AUDIOIN");
    audioOUTObject = PyUnicode_FromString("AUDIOOUT");
    pdAudio = PyUnicode_FromString("AUDIO");
    Py4pd_OutLoopString = PyUnicode_FromString(OUT_string);
    Py4pd_ClearLoopString = PyUnicode_FromString(CLEAR_string);

    PyModule_AddObject(py4pdmodule, "SAMPLERATE", puredata_samplerate);
    PyModule_AddObject(py4pdmodule, "VECSIZE", puredata_vecsize);
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
    return py4pdmodule;
}

// =============================
