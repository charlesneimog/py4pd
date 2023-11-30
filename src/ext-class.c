#include "py4pd.h"

// ============================================================
// ===================== PDOBJECT CLASS =======================
// ============================================================
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
} py4pdNewObject;

// ============================================================
static int Py4pdNewObj_init(py4pdNewObject *self, PyObject *args,
                            PyObject *kwds) {
  (void)self;
  static char *keywords[] = {"param_str", NULL};
  char *param_str;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", keywords, &param_str)) {
    return -1;
  }
  PyObject *objectName = PyUnicode_FromString(param_str);
  if (!objectName) {
    PyErr_SetString(PyExc_TypeError, "Object name not valid");
    return -1;
  }
  self->objName = PyUnicode_AsUTF8(objectName);
  self->objType = PY4PD_NORMALOBJ;
  return 0;
}

// ============================================================
static PyObject *Py4pdNewObj_GetType(py4pdNewObject *self) {
  return PyLong_FromLong(self->objType);
}

static int Py4pdNewObj_SetType(py4pdNewObject *self, PyObject *value) {
  int typeValue = PyLong_AsLong(value);
  if (typeValue < 0 || typeValue > 3) {
    PyErr_SetString(PyExc_TypeError,
                    "Object type not supported, check the value");
    return -1;
  }
  self->objType = typeValue;
  return 0;
}
// ============================================================
static PyObject *Py4pdNewObj_GetName(py4pdNewObject *self) {
  return PyUnicode_FromString(self->objName);
}

// ++++++++++
static int Py4pdNewObj_SetName(py4pdNewObject *self, PyObject *value) {
  PyObject *objectName = PyUnicode_FromString(PyUnicode_AsUTF8(value));
  if (!objectName) {
    PyErr_SetString(PyExc_TypeError, "Object name not valid");
    return -1;
  }
  self->objName = PyUnicode_AsUTF8(objectName);
  Py_DECREF(objectName);
  return 0;
}

// ============================================================
static PyObject *Py4pdNewObj_GetPlayable(py4pdNewObject *self) {
  return PyLong_FromLong(self->objIsPlayable);
}

// ++++++++++
static int Py4pdNewObj_SetPlayable(py4pdNewObject *self, PyObject *value) {
  int playableValue = PyObject_IsTrue(value);
  if (playableValue < 0 || playableValue > 1) {
    PyErr_SetString(PyExc_TypeError,
                    "Playable value not supported, check the value");
    return -1;
  }
  self->objIsPlayable = playableValue;
  return 0;
}

// ============================================================
static PyObject *Py4pdNewObj_GetImage(py4pdNewObject *self) {
  return PyUnicode_FromString(self->objImage);
}

// ++++++++++
static int Py4pdNewObj_SetImage(py4pdNewObject *self, PyObject *value) {
  const char *imageValue = PyUnicode_AsUTF8(value);
  if (!imageValue) {
    PyErr_SetString(PyExc_TypeError, "Image path not valid");
    return -1;
  }
  self->objImage = imageValue;
  return 0;
}

// ============================================================
static PyObject *Py4pdNewObj_GetOutputPyObjs(py4pdNewObject *self) {
  return PyLong_FromLong(self->objOutPyObjs);
}

// ++++++++++
static int Py4pdNewObj_SetOutputPyObjs(py4pdNewObject *self, PyObject *value) {
  int outputPyObjsValue = PyObject_IsTrue(value);
  if (!outputPyObjsValue) {
    PyErr_SetString(PyExc_TypeError, "Image path not valid");
    return -1;
  }
  self->objOutPyObjs = outputPyObjsValue;
  return 0;
}

// ============================================================
static PyObject *Py4pdNewObj_GetFigSize(py4pdNewObject *self) {
  return PyLong_FromLong(self->objIsPlayable);
}

// ++++++++++
static int Py4pdNewObj_SetFigSize(py4pdNewObject *self, PyObject *value) {
  // see if object value is a tuple with two integers
  if (!PyTuple_Check(value)) {
    PyErr_SetString(PyExc_TypeError,
                    "Figure size must be a tuple with two integers");
    return -1; // Error handling
  }
  if (PyTuple_Size(value) != 2) {
    PyErr_SetString(PyExc_TypeError,
                    "Figure size must be a tuple with two integers");
    return -1; // Error handling
  }
  PyObject *width = PyTuple_GetItem(value, 0);
  PyObject *height = PyTuple_GetItem(value, 1);
  if (!PyLong_Check(width) || !PyLong_Check(height)) {
    PyErr_SetString(PyExc_TypeError,
                    "Figure size must be a tuple with two integers");
    return -1; // Error handling
  }
  self->objFigSize = value;

  return 0; // Success
}

// ============================================================
static PyObject *Py4pdNewObj_GetNoOutlets(py4pdNewObject *self) {
  if (self->noOutlets) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

// ++++++++++
static int Py4pdNewObj_SetNoOutlets(
    py4pdNewObject *self,
    PyObject *value) { // see if object value is a tuple with two integers
  int outpuNoOutlets = PyObject_IsTrue(value);
  self->noOutlets = outpuNoOutlets;
  return 0; // Success
}

// ============================================================
static PyObject *Py4pdNewObj_GetRequireNofOuts(py4pdNewObject *self) {
  if (self->requireNofOutlets) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

// ++++++++++
static int Py4pdNewObj_SetRequireNofOuts(
    py4pdNewObject *self,
    PyObject *value) { // see if object value is a tuple with two integers
  int outpuNoOutlets = PyObject_IsTrue(value);
  self->requireNofOutlets = outpuNoOutlets;
  return 0; // Success
}

// ============================================================
static PyObject *Py4pdNewObj_GetAuxOutNumbers(py4pdNewObject *self) {
  return PyLong_FromLong(self->auxOutlets);
}

// ++++++++++
static int Py4pdNewObj_SetAuxOutNumbers(
    py4pdNewObject *self,
    PyObject *value) { // see if object value is a tuple with two integers
  if (!PyLong_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "Auxiliary outlets must be an integer");
    return -1; // Error handling
  }
  self->auxOutlets = PyLong_AsLong(value);
  return 0; // Success
}

// ============================================================
static PyObject *Py4pdNewObj_GetIgnoreNone(py4pdNewObject *self) {
  if (self->ignoreNoneOutputs) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

// ++++++++++
static int Py4pdNewObj_SetIgnoreNone(
    py4pdNewObject *self,
    PyObject *value) { // see if object value is a tuple with two integers
  int ignoreNone = PyObject_IsTrue(value);
  self->ignoreNoneOutputs = ignoreNone;
  return 0; // Success
}

// ========================== GET/SET ==========================
static PyGetSetDef Py4pdNewObj_GetSet[] = {
    {"type", (getter)Py4pdNewObj_GetType, (setter)Py4pdNewObj_SetType,
     "pd.NORMALOBJ, Type attribute", NULL},
    {"name", (getter)Py4pdNewObj_GetName, (setter)Py4pdNewObj_SetName,
     "Name attribute", NULL},
    {"playable", (getter)Py4pdNewObj_GetPlayable,
     (setter)Py4pdNewObj_SetPlayable, "Playable attribute", NULL},
    {"figsize", (getter)Py4pdNewObj_GetFigSize, (setter)Py4pdNewObj_SetFigSize,
     "Figure size attribute", NULL},
    {"image", (getter)Py4pdNewObj_GetImage, (setter)Py4pdNewObj_SetImage,
     "Image patch", NULL},
    {"pyout", (getter)Py4pdNewObj_GetOutputPyObjs,
     (setter)Py4pdNewObj_SetOutputPyObjs, "Output or not PyObjs", NULL},
    {"no_outlet", (getter)Py4pdNewObj_GetNoOutlets,
     (setter)Py4pdNewObj_SetNoOutlets, "Number of outlets", NULL},
    {"require_n_of_outlets", (getter)Py4pdNewObj_GetRequireNofOuts,
     (setter)Py4pdNewObj_SetRequireNofOuts, "Require number of outlets", NULL},
    {"n_extra_outlets", (getter)Py4pdNewObj_GetAuxOutNumbers,
     (setter)Py4pdNewObj_SetAuxOutNumbers, "Number of auxiliary outlets", NULL},
    {"ignore_none", (getter)Py4pdNewObj_GetIgnoreNone,
     (setter)Py4pdNewObj_SetIgnoreNone, "Ignore None outputs", NULL},
    {NULL, NULL, NULL, NULL, NULL} /* Sentinel */
};

// =============================================================
static PyObject *Py4pdNewObj_Method_AddObj(py4pdNewObject *self,
                                           PyObject *args) {
  (void)args;

  PyObject *objConfigDict = PyDict_New();
  const char *objectName;
  objectName = self->objName;

  PyObject *Py_ObjType = PyLong_FromLong(self->objType);
  PyDict_SetItemString(objConfigDict, "py4pdOBJType", Py_ObjType);
  Py_DECREF(Py_ObjType);
  t_class *objClass;

  if (self->objType == PY4PD_NORMALOBJ) {
    objClass = class_new(gensym(objectName), (t_newmethod)Py4pdLib_NewObj,
                         (t_method)Py4pdLib_FreeObj, sizeof(t_py),
                         CLASS_DEFAULT, A_GIMME, 0);
  } else if (self->objType == PY4PD_VISOBJ) {
    objClass = class_new(gensym(objectName), (t_newmethod)Py4pdLib_NewObj,
                         (t_method)Py4pdLib_FreeObj, sizeof(t_py),
                         CLASS_DEFAULT, A_GIMME, 0);
  } else if (self->objType == PY4PD_AUDIOINOBJ) {
    objClass = class_new(gensym(objectName), (t_newmethod)Py4pdLib_NewObj,
                         (t_method)Py4pdLib_FreeObj, sizeof(t_py),
                         CLASS_MULTICHANNEL, A_GIMME, 0);
  } else if (self->objType == PY4PD_AUDIOOUTOBJ) {
    objClass = class_new(gensym(objectName), (t_newmethod)Py4pdLib_NewObj,
                         (t_method)Py4pdLib_FreeObj, sizeof(t_py),
                         CLASS_MULTICHANNEL, A_GIMME, 0);
  } else if (self->objType == PY4PD_AUDIOOBJ) {
    objClass = class_new(gensym(objectName), (t_newmethod)Py4pdLib_NewObj,
                         (t_method)Py4pdLib_FreeObj, sizeof(t_py),
                         CLASS_MULTICHANNEL, A_GIMME, 0);
  } else {
    PyErr_SetString(PyExc_TypeError,
                    "Object type not supported, check the spelling");
    return NULL;
  }

  PyObject *Py_ClassLocal = PyLong_FromVoidPtr(objClass);
  PyDict_SetItemString(objConfigDict, "py4pdOBJ_CLASS", Py_ClassLocal);
  Py_DECREF(Py_ClassLocal);

  if (self->objFigSize != NULL) {
    PyObject *Py_Width = PyTuple_GetItem(self->objFigSize, 0);
    PyDict_SetItemString(objConfigDict, "py4pdOBJwidth", Py_Width);
    Py_DECREF(Py_Width);
    PyObject *Py_Height = PyTuple_GetItem(self->objFigSize, 1);
    PyDict_SetItemString(objConfigDict, "py4pdOBJheight", Py_Height);
    Py_DECREF(Py_Height);
  } else {
    PyObject *Py_Width = PyLong_FromLong(250);
    PyDict_SetItemString(objConfigDict, "py4pdOBJwidth", Py_Width);
    Py_DECREF(Py_Width);
    PyObject *Py_Height = PyLong_FromLong(250);
    PyDict_SetItemString(objConfigDict, "py4pdOBJheight", Py_Height);
    Py_DECREF(Py_Height);
  }

  PyObject *Py_Playable = PyLong_FromLong(self->objIsPlayable);
  PyDict_SetItemString(objConfigDict, "py4pdOBJPlayable", Py_Playable);
  Py_DECREF(Py_Playable);

  if (self->objImage != NULL) {
    PyObject *Py_GifImage = PyUnicode_FromString(self->objImage);
    PyDict_SetItemString(objConfigDict, "py4pdOBJGif", Py_GifImage);
    Py_DECREF(Py_GifImage);
  }

  PyObject *Py_ObjOuts = PyLong_FromLong(self->objOutPyObjs);
  PyDict_SetItemString(objConfigDict, "py4pdOBJpyout", Py_ObjOuts);
  Py_DECREF(Py_ObjOuts);

  PyObject *Py_NoOutlet = PyLong_FromLong(self->noOutlets);
  PyDict_SetItemString(objConfigDict, "py4pdOBJnooutlet", Py_NoOutlet);
  Py_DECREF(Py_NoOutlet);

  PyObject *Py_RequireOutletN = PyLong_FromLong(self->requireNofOutlets);
  PyDict_SetItemString(objConfigDict, "py4pdOBJrequireoutletn",
                       Py_RequireOutletN);
  Py_DECREF(Py_RequireOutletN);

  PyObject *Py_auxOutlets = PyLong_FromLong(self->auxOutlets);
  PyDict_SetItemString(objConfigDict, "py4pdAuxOutlets", Py_auxOutlets);
  Py_DECREF(Py_auxOutlets);

  PyObject *Py_ObjName = PyUnicode_FromString(objectName);
  PyDict_SetItemString(objConfigDict, "py4pdOBJname", Py_ObjName);
  Py_DECREF(Py_ObjName);

  PyObject *Py_IgnoreNoneReturn = PyLong_FromLong(self->ignoreNoneOutputs);
  PyDict_SetItemString(objConfigDict, "py4pdOBJIgnoreNone",
                       Py_IgnoreNoneReturn);
  Py_DECREF(Py_IgnoreNoneReturn);

  PyObject *objectDict = PyDict_New();
  PyDict_SetItemString(objectDict, objectName, objConfigDict);
  PyObject *py4pd_capsule = PyCapsule_New(objectDict, objectName, NULL);
  char py4pd_objectName[MAXPDSTRING];
  sprintf(py4pd_objectName, "py4pd_ObjectDict_%s", objectName);

  PyObject *pdModule = PyImport_ImportModule("pd");
  PyModule_AddObject(pdModule, py4pd_objectName, py4pd_capsule);
  Py_DECREF(pdModule);

  Py_RETURN_TRUE;
}

// ========================== METHODS ==========================
static PyMethodDef Py4pdNewObj_methods[] = {
    {"add_object", (PyCFunction)Py4pdNewObj_Method_AddObj, METH_NOARGS,
     "Get the type attribute"},

    // {"addmethod_float"
    // {"addmethod_symbol"
    // {"addmethod_list"
    // {"addmethod_anything"
    // {"addmethod_bang"
    // {"addmethod"

    //     EXTERN void class_addmethod(t_class *c, t_method fn, t_symbol *sel,
    //     t_atomtype arg1, ...);
    // EXTERN void class_addbang(t_class *c, t_method fn);
    // EXTERN void class_addpointer(t_class *c, t_method fn);
    // EXTERN void class_doaddfloat(t_class *c, t_method fn);
    // EXTERN void class_addsymbol(t_class *c, t_method fn);
    // EXTERN void class_addlist(t_class *c, t_method fn);
    // EXTERN void class_addanything(t_class *c, t_method fn);

    {NULL, NULL, 0, NULL} // Sentinel
};

// ========================== CLASS ============================
PyTypeObject Py4pdNewObj_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pd.NewObject",
    .tp_doc = "It creates new PureData objects",
    .tp_basicsize = sizeof(py4pdNewObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Py4pdNewObj_init,
    .tp_methods = Py4pdNewObj_methods,
    .tp_getset = Py4pdNewObj_GetSet, // Add the GetSet descriptors here
};
