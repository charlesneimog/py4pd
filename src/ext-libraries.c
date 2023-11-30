#include "py4pd.h"

// ===========================================
// ================= PUREDATA ================
// ===========================================

static t_class *py4pdInlets_proxy_class;
void Py4pdLib_Bang(t_py *x);

// ===========================================
void Py4pdLib_ReloadObject(t_py *x) {
  char *script_filename = strdup(x->pScriptName->s_name);
  const char *ScriptFileName = Py4pdUtils_GetFilename(script_filename);
  free(script_filename);

  PyObject *pModule = PyImport_ImportModule(ScriptFileName);
  if (pModule == NULL) {
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
    PyObject *pstr = PyObject_Str(pvalue);
    pd_error(x, "[Python] %s", PyUnicode_AsUTF8(pstr));
    Py_DECREF(pstr);
    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptraceback);
    PyErr_Clear();
    return;
  }
  PyObject *pModuleReloaded = PyImport_ReloadModule(pModule);
  if (pModuleReloaded == NULL) {
    pd_error(x, "[Python] Failed to reload module");
    return;
  }
  x->pFunction = PyObject_GetAttrString(pModuleReloaded, x->pFuncName->s_name);
  if (x->pFunction == NULL) {
    pd_error(x, "[Python] Failed to get function");
    return;
  } else {
    post("[Python] Function reloaded");
    x->audioError = 0;
  }
  return;
}

// ===========================================
void Py4pdLib_Py4pdObjPicSave(t_gobj *z, t_binbuf *b) {

  t_py *x = (t_py *)z;
  if (x->visMode) {
    binbuf_addv(b, "ssii", gensym("#X"), gensym("obj"), x->obj.te_xpix,
                x->obj.te_ypix);
    binbuf_addbinbuf(b, ((t_py *)x)->obj.te_binbuf);
    int objAtomsCount = binbuf_getnatom(((t_py *)x)->obj.te_binbuf);
    if (objAtomsCount == 1) {
      binbuf_addv(b, "ii", x->width, x->height);
    }
    binbuf_addsemi(b);
  }
  return;
}

// ===========================================
void Py4pdLib_Click(t_py *x) {
  PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(x->pFunction);
  int line = PyCode_Addr2Line(code, 0);
  char command[MAXPDSTRING];
  Py4pdUtils_GetEditorCommand(x, command, line);
  Py4pdUtils_ExecuteSystemCommand(command);
  return;
}

// ===========================================
void Py4pdLib_SetKwargs(t_py *x, t_symbol *s, int ac, t_atom *av) {
  (void)s;
  t_symbol *key;

  if (av[0].a_type != A_SYMBOL) {
    pd_error(x, "The first argument of the message 'kwargs' must be a symbol");
    return;
  }
  if (ac < 2) {
    pd_error(x, "You need to specify a value for the key");
    return;
  }
  key = atom_getsymbolarg(0, ac, av);

  if (x->kwargsDict == NULL)
    x->kwargsDict = PyDict_New();

  PyObject *oldItemFromDict = PyDict_GetItemString(x->kwargsDict, s->s_name);
  if (oldItemFromDict != NULL) {
    PyDict_DelItemString(x->kwargsDict, s->s_name);
    Py_DECREF(oldItemFromDict);
  }

  if (ac == 2) {
    if (av[1].a_type == A_FLOAT) {
      int isInt = atom_getintarg(1, ac, av) == atom_getfloatarg(1, ac, av);
      if (isInt)
        PyDict_SetItemString(x->kwargsDict, key->s_name,
                             PyLong_FromLong(atom_getintarg(1, ac, av)));
      else
        PyDict_SetItemString(x->kwargsDict, key->s_name,
                             PyFloat_FromDouble(atom_getfloatarg(1, ac, av)));

    } else if (av[1].a_type == A_SYMBOL)
      PyDict_SetItemString(
          x->kwargsDict, key->s_name,
          PyUnicode_FromString(atom_getsymbolarg(1, ac, av)->s_name));
    else {
      pd_error(x, "The third argument of the message 'kwargs' must be a "
                  "symbol or a float");
      return;
    }
  } else if (ac > 2) {
    PyObject *pyInletValue = PyList_New(ac - 1);
    for (int i = 1; i < ac; i++) {
      if (av[i].a_type == A_FLOAT) {
        int isInt = atom_getintarg(i, ac, av) == atom_getfloatarg(i, ac, av);
        if (isInt)
          PyList_SetItem(pyInletValue, i - 1,
                         PyLong_FromLong(atom_getintarg(i, ac, av)));
        else
          PyList_SetItem(pyInletValue, i - 1,
                         PyFloat_FromDouble(atom_getfloatarg(i, ac, av)));
      } else if (av[i].a_type == A_SYMBOL)
        PyList_SetItem(
            pyInletValue, i - 1,
            PyUnicode_FromString(atom_getsymbolarg(i, ac, av)->s_name));
    }
    PyDict_SetItemString(x->kwargsDict, key->s_name, pyInletValue);
  }
  return;
}

// ===========================================
static int Py4pdLib_CreateObjInlets(PyObject *function, t_py *x, int argc,
                                    t_atom *argv) {
  (void)function;
  t_pd **py4pdInlet_proxies;
  int i;
  int pyFuncArgs = x->pArgsCount - 1;

  PyObject *defaults = PyObject_GetAttrString(
      function, "__defaults__"); // TODO:, WHERE CLEAR THIS?
  int defaultsCount = PyTuple_Size(defaults);

  if (x->use_pArgs && defaultsCount > 0) {
    pd_error(x, "[py4pd] You can't use *args and defaults at the same time");
    return -1;
  }
  int indexWhereStartDefaults = x->pArgsCount - defaultsCount;

  t_py4pd_pValue *PyPtrValueMain =
      (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
  PyPtrValueMain->objectsUsing = 0;
  PyPtrValueMain->pdout = 0;
  PyPtrValueMain->objOwner = x->objName;
  if (indexWhereStartDefaults == 0) {
    PyPtrValueMain->pValue = PyTuple_GetItem(defaults, 0);
    x->pyObjArgs[0] = PyPtrValueMain;
  } else {
    Py_INCREF(Py_None);
    PyPtrValueMain->pValue = Py_None;
    x->pyObjArgs[0] = PyPtrValueMain;
  }

  if (pyFuncArgs != 0) {
    py4pdInlet_proxies =
        (t_pd **)getbytes((pyFuncArgs + 1) * sizeof(*py4pdInlet_proxies));
    for (i = 0; i < pyFuncArgs; i++) {
      py4pdInlet_proxies[i] = pd_new(py4pdInlets_proxy_class);
      t_py4pdInlet_proxy *y = (t_py4pdInlet_proxy *)py4pdInlet_proxies[i];
      y->p_master = x;
      y->inletIndex = i + 1;
      inlet_new((t_object *)x, (t_pd *)y, 0, 0);
      t_py4pd_pValue *PyPtrValue =
          (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
      PyPtrValue->objectsUsing = 0;
      PyPtrValue->pdout = 0;
      PyPtrValue->objOwner = x->objName;
      if (i + 1 >= indexWhereStartDefaults) {
        PyPtrValue->pValue =
            PyTuple_GetItem(defaults, (i + 1) - indexWhereStartDefaults);
      } else {
        PyPtrValue->pValue = Py_None;
        Py_INCREF(Py_None);
      }
      x->pyObjArgs[i + 1] = PyPtrValue;
    }
    int argNumbers = x->pArgsCount;

    for (i = 0; i < argNumbers; i++) {
      if (i <= argc) {
        if (argv[i].a_type == A_FLOAT) {
          int isInt =
              atom_getintarg(i, argc, argv) == atom_getfloatarg(i, argc, argv);
          if (isInt) {
            x->pyObjArgs[i]->pValue =
                PyLong_FromLong(atom_getfloatarg(i, argc, argv));
          } else {
            x->pyObjArgs[i]->pValue =
                PyFloat_FromDouble(atom_getfloatarg(i, argc, argv));
          }
        }

        else if (argv[i].a_type == A_SYMBOL) {
          if (strcmp(atom_getsymbolarg(i, argc, argv)->s_name, "None") == 0) {
            Py_INCREF(Py_None);
            x->pyObjArgs[i]->pValue = Py_None;
          } else {
            x->pyObjArgs[i]->pValue =
                PyUnicode_FromString(atom_getsymbolarg(i, argc, argv)->s_name);
          }
        } else if (x->pyObjArgs[i]->pValue == NULL) {
          Py_INCREF(Py_None);
          x->pyObjArgs[i]->pValue = Py_None;
        }
      } else if (x->pyObjArgs[i]->pValue == NULL) {
        Py_INCREF(Py_None);
        x->pyObjArgs[i]->pValue = Py_None;
      }
    }
  }
  return 0;
}

// ===========================================
static void Py4pdLib_Audio2PdAudio(t_py *x, PyObject *pValue,
                                   t_sample *audioOut, int numChannels, int n) {
  if (pValue != NULL) {
    if (PyArray_Check(pValue)) {

      PyArrayObject *pArray = PyArray_GETCONTIGUOUS((PyArrayObject *)pValue);
      PyArray_Descr *pArrayType = PyArray_DESCR(pArray); // double or float
      int arrayLength = PyArray_SIZE(pArray);
      int returnedChannels = PyArray_DIM(pArray, 0);

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

// +++++++++++++
// ++ METHODS ++
// +++++++++++++
void Py4pdLib_ProxyPointer(t_py4pdInlet_proxy *x, t_symbol *s, t_gpointer *gp) {
  (void)s;
  t_py *py4pd = (t_py *)x->p_master;
  t_py4pd_pValue *pArg;
  pArg = (t_py4pd_pValue *)gp;
  if (!pArg->pdout)
    Py_DECREF(py4pd->pyObjArgs[x->inletIndex]->pValue);

  Py4pdUtils_CopyPy4pdValueStruct(pArg, py4pd->pyObjArgs[x->inletIndex]);

  if (!pArg->pdout)
    Py_INCREF(py4pd->pyObjArgs[x->inletIndex]->pValue);

  return;
}

// =============================================
void Py4pdLib_Pointer(t_py *x, t_symbol *s, t_gpointer *gp) {
  (void)s;

  t_py4pd_pValue *pArg;
  pArg = (t_py4pd_pValue *)gp;
  pArg->objectsUsing++;

  if (x->objType > PY4PD_AUDIOINOBJ) {
    pArg = (t_py4pd_pValue *)gp;
    if (!pArg->pdout)
      Py_DECREF(x->pyObjArgs[0]->pValue);

    Py4pdUtils_CopyPy4pdValueStruct(pArg, x->pyObjArgs[0]);

    if (!pArg->pdout)
      Py_INCREF(x->pyObjArgs[0]->pValue);
    x->audioError = 0;
    return;
  }

  if (x->objType > PY4PD_AUDIOINOBJ)
    return;

  PyObject *pArgs = PyTuple_New(x->pArgsCount);
  PyTuple_SetItem(pArgs, 0, pArg->pValue);
  Py_INCREF(pArg->pValue);

  for (int i = 1; i < x->pArgsCount; i++) {
    PyTuple_SetItem(pArgs, i, x->pyObjArgs[i]->pValue);
    Py_INCREF(x->pyObjArgs[i]->pValue);
  }

  Py4pdUtils_RunPy(x, pArgs, x->kwargsDict);
  Py_DECREF(pArgs);

  return;
}

// =====================================
void Py4pdLib_ProxyAnything(t_py4pdInlet_proxy *x, t_symbol *s, int ac,
                            t_atom *av) {
  (void)s;
  t_py *py4pd = (t_py *)x->p_master;
  PyObject *pyInletValue = NULL;

  if (ac == 0)
    pyInletValue = PyUnicode_FromString(s->s_name);
  else if ((s == gensym("list") || s == gensym("anything")) && ac > 1) {
    pyInletValue = PyList_New(ac);
    for (int i = 0; i < ac; i++) {
      if (av[i].a_type == A_FLOAT) {
        int isInt = atom_getintarg(i, ac, av) == atom_getfloatarg(i, ac, av);
        if (isInt)
          PyList_SetItem(pyInletValue, i,
                         PyLong_FromLong(atom_getintarg(i, ac, av)));
        else
          PyList_SetItem(pyInletValue, i,
                         PyFloat_FromDouble(atom_getfloatarg(i, ac, av)));
      } else if (av[i].a_type == A_SYMBOL)
        PyList_SetItem(
            pyInletValue, i,
            PyUnicode_FromString(atom_getsymbolarg(i, ac, av)->s_name));
    }
  } else if ((s == gensym("float") || s == gensym("symbol")) && ac == 1) {
    if (av[0].a_type == A_FLOAT) {
      int isInt = atom_getintarg(0, ac, av) == atom_getfloatarg(0, ac, av);
      if (isInt)
        pyInletValue = PyLong_FromLong(atom_getintarg(0, ac, av));
      else
        pyInletValue = PyFloat_FromDouble(atom_getfloatarg(0, ac, av));
    } else if (av[0].a_type == A_SYMBOL)
      pyInletValue = PyUnicode_FromString(atom_getsymbolarg(0, ac, av)->s_name);
  } else {
    pyInletValue = PyList_New(ac + 1);
    PyList_SetItem(pyInletValue, 0, PyUnicode_FromString(s->s_name));
    for (int i = 0; i < ac; i++) {
      if (av[i].a_type == A_FLOAT) {
        int isInt = atom_getintarg(i, ac, av) == atom_getfloatarg(i, ac, av);
        if (isInt)
          PyList_SetItem(pyInletValue, i + 1,
                         PyLong_FromLong(atom_getintarg(i, ac, av)));
        else
          PyList_SetItem(pyInletValue, i + 1,
                         PyFloat_FromDouble(atom_getfloatarg(i, ac, av)));
      } else if (av[i].a_type == A_SYMBOL)
        PyList_SetItem(
            pyInletValue, i + 1,
            PyUnicode_FromString(atom_getsymbolarg(i, ac, av)->s_name));
    }
  }
  if (!py4pd->pyObjArgs[x->inletIndex]->pdout)
    Py_DECREF(py4pd->pyObjArgs[x->inletIndex]->pValue);

  py4pd->pyObjArgs[x->inletIndex]->objectsUsing = 0;
  py4pd->pyObjArgs[x->inletIndex]->pdout = 0;
  py4pd->pyObjArgs[x->inletIndex]->objOwner = py4pd->objName;
  py4pd->pyObjArgs[x->inletIndex]->pValue = pyInletValue;

  if (py4pd->objType == PY4PD_AUDIOOBJ || py4pd->objType == PY4PD_AUDIOINOBJ) {
    py4pd->audioError = 0;
    return;
  }

  return;
}

// =====================================
void Py4pdLib_Anything(t_py *x, t_symbol *s, int ac, t_atom *av) {

  if (x->pFunction == NULL) {
    pd_error(x, "[%s]: No function defined", x->objName->s_name);
    return;
  }
  if (s == gensym("bang")) {
    Py4pdLib_Bang(x);
    return;
  }

  PyObject *pyInletValue = NULL;
  if (ac == 0)
    pyInletValue = PyUnicode_FromString(s->s_name);
  else if ((s == gensym("list") || s == gensym("anything")) && ac > 1) {
    pyInletValue = PyList_New(ac);
    for (int i = 0; i < ac; i++) {
      if (av[i].a_type == A_FLOAT) {
        int isInt = atom_getintarg(i, ac, av) == atom_getfloatarg(i, ac, av);
        if (isInt)
          PyList_SetItem(pyInletValue, i,
                         PyLong_FromLong(atom_getintarg(i, ac, av)));
        else
          PyList_SetItem(pyInletValue, i,
                         PyFloat_FromDouble(atom_getfloatarg(i, ac, av)));
      } else if (av[i].a_type == A_SYMBOL)
        PyList_SetItem(
            pyInletValue, i,
            PyUnicode_FromString(atom_getsymbolarg(i, ac, av)->s_name));
    }
  } else if ((s == gensym("float") || s == gensym("symbol")) && ac == 1) {
    if (av[0].a_type == A_FLOAT) {
      int isInt = atom_getintarg(0, ac, av) == atom_getfloatarg(0, ac, av);
      if (isInt)
        pyInletValue = PyLong_FromLong(atom_getintarg(0, ac, av));
      else
        pyInletValue = PyFloat_FromDouble(atom_getfloatarg(0, ac, av));
    } else if (av[0].a_type == A_SYMBOL)
      pyInletValue = PyUnicode_FromString(atom_getsymbolarg(0, ac, av)->s_name);
  } else {
    pyInletValue = PyList_New(ac + 1);
    PyList_SetItem(pyInletValue, 0, PyUnicode_FromString(s->s_name));
    for (int i = 0; i < ac; i++) {
      if (av[i].a_type == A_FLOAT) {
        int isInt = atom_getintarg(i, ac, av) == atom_getfloatarg(i, ac, av);
        if (isInt)
          PyList_SetItem(pyInletValue, i + 1,
                         PyLong_FromLong(atom_getintarg(i, ac, av)));
        else
          PyList_SetItem(pyInletValue, i + 1,
                         PyFloat_FromDouble(atom_getfloatarg(i, ac, av)));
      } else if (av[i].a_type == A_SYMBOL)
        PyList_SetItem(
            pyInletValue, i + 1,
            PyUnicode_FromString(atom_getsymbolarg(i, ac, av)->s_name));
    }
  }

  if (x->objType == PY4PD_AUDIOOUTOBJ) {
    x->pyObjArgs[0]->pValue = pyInletValue;
    x->audioError = 0;
    return;
  }

  if (x->objType > PY4PD_AUDIOINOBJ)
    return;

  PyObject *pArgs = PyTuple_New(x->pArgsCount);
  PyTuple_SetItem(pArgs, 0, pyInletValue);

  for (int i = 1; i < x->pArgsCount; i++) {
    PyTuple_SetItem(pArgs, i, x->pyObjArgs[i]->pValue);
    Py_INCREF(x->pyObjArgs[i]->pValue); // This keep the reference.
  }

  if (x->objType > PY4PD_VISOBJ) {
    Py_INCREF(pyInletValue);
    x->pyObjArgs[0]->objectsUsing = 0;
    x->pyObjArgs[0]->pdout = 0;
    x->pyObjArgs[0]->objOwner = x->objName;
    x->pyObjArgs[0]->pValue = pyInletValue;
    x->pArgTuple = pArgs;
    Py_INCREF(x->pArgTuple);
    return;
  }

  Py4pdUtils_RunPy(x, pArgs, x->kwargsDict);
  Py4pdUtils_DECREF(pArgs);
  PyErr_Clear();
  return;
}

// =====================================
void Py4pdLib_Bang(t_py *x) {
  if (x->pArgsCount != 0) {
    pd_error(x, "[%s] Bang can be used only with no arguments Function.",
             x->objName->s_name);
    return;
  }
  if (x->pFunction == NULL) {
    pd_error(x, "[%s] Function not defined", x->objName->s_name);
    return;
  }
  if (x->audioOutput) {
    return;
  }
  PyObject *pArgs = PyTuple_New(0);
  Py4pdUtils_RunPy(x, pArgs, x->kwargsDict);
  Py_DECREF(pArgs);
}

// =====================================
//             +++++++++++            //
//             ++ AUDIO ++            //
//             +++++++++++            //
// =====================================
t_int *Py4pdLib_AudioINPerform(t_int *w) {
  t_py *x = (t_py *)(w[1]);

  if (x->audioError || x->numpyImported == 0)
    return (w + 4);

  t_sample *in = (t_sample *)(w[2]);
  int n = (int)(w[3]);
  x->vectorSize = n;
  int numChannels = n / x->vectorSize;
  npy_intp dims[] = {numChannels, x->vectorSize};
  PyObject *pAudio = PyArray_SimpleNewFromData(
      2, dims, NPY_FLOAT, in); // TODO: this should be DOUBLE in pd64
  if (x->pArgTuple == NULL)
    return (w + 4);
  PyTuple_SetItem(x->pArgTuple, 0, pAudio);
  for (int i = 1; i < x->pArgsCount; i++) {
    PyTuple_SetItem(x->pArgTuple, i, x->pyObjArgs[i]->pValue);
    Py_INCREF(x->pyObjArgs[i]->pValue);
  }
  Py4pdUtils_RunPy(x, x->pArgTuple, x->kwargsDict);
  return (w + 4);
}

// =====================================
t_int *Py4pdLib_AudioOUTPerform(t_int *w) {
  t_py *x = (t_py *)(w[1]);

  if (x->audioError || x->numpyImported == 0)
    return (w + 4);

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
  Py4pdLib_Audio2PdAudio(x, pValue, audioOut, numChannels, n);
  Py_XDECREF(pValue);
  return (w + 4);
}

// =====================================
t_int *Py4pdLib_AudioPerform(t_int *w) {
  t_py *x = (t_py *)(w[1]);
  if (x->audioError || x->numpyImported == 0)
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
  Py4pdLib_Audio2PdAudio(x, pValue, audioOut, numChannels, n);

  Py_XDECREF(pValue);

  return (w + 5);
}

// =====================================
static void Py4pdLib_Dsp(t_py *x, t_signal **sp) {
  if (_import_array() != 0) {
    x->numpyImported = 0;
    pd_error(x, "[py4pd] Failed to import numpy");
  } else {
    x->numpyImported = 1;
  }

  if (x->objType == PY4PD_AUDIOINOBJ) {
    x->nChs = sp[0]->s_nchans;
    x->vectorSize = sp[0]->s_n;
    dsp_add(Py4pdLib_AudioINPerform, 3, x, sp[0]->s_vec, PY4PDSIGTOTAL(sp[0]));
  } else if (x->objType == PY4PD_AUDIOOUTOBJ) {
    x->vectorSize = sp[0]->s_n;
    signal_setmultiout(&sp[0], x->nChs);
    dsp_add(Py4pdLib_AudioOUTPerform, 3, x, sp[0]->s_vec, PY4PDSIGTOTAL(sp[0]));
  } else if (x->objType == PY4PD_AUDIOOBJ) {
    x->nChs = sp[0]->s_nchans;
    x->vectorSize = sp[0]->s_n;
    signal_setmultiout(&sp[1], sp[0]->s_nchans);
    dsp_add(Py4pdLib_AudioPerform, 4, x, sp[0]->s_vec, sp[1]->s_vec,
            PY4PDSIGTOTAL(sp[0]));
  }
}

// ================
// == CREATE OBJ ==
// ================
void *Py4pdLib_NewObj(t_symbol *s, int argc, t_atom *argv) {
  const char *objectName = s->s_name;
  char py4pd_objectName[MAXPDSTRING];
  snprintf(py4pd_objectName, sizeof(py4pd_objectName), "py4pd_ObjectDict_%s",
           objectName);

  PyObject *pd_module = PyImport_ImportModule("pd");
  PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, py4pd_objectName);
  PyObject *PdDictCapsule = PyCapsule_GetPointer(py4pd_capsule, objectName);
  if (PdDictCapsule == NULL) {
    pd_error(NULL, "Error: PdDictCapsule is NULL, please report!");
    Py_DECREF(pd_module);
    Py_DECREF(py4pd_capsule);
    return NULL;
  }
  PyObject *PdDict = PyDict_GetItemString(PdDictCapsule, objectName);
  if (PdDict == NULL) {
    pd_error(NULL, "Error: PdDict is NULL");
    Py_DECREF(pd_module);
    Py_DECREF(py4pd_capsule);
    return NULL;
  }

  PyObject *PY_objectClass = PyDict_GetItemString(PdDict, "py4pdOBJ_CLASS");
  if (PY_objectClass == NULL) {
    pd_error(NULL, "Error: object Class is NULL");
    Py_DECREF(pd_module);
    Py_DECREF(py4pd_capsule);
    return NULL;
  }

  // post("All important things work!");
  PyObject *ignoreOnNone = PyDict_GetItemString(PdDict, "py4pdOBJIgnoreNone");
  PyObject *playable = PyDict_GetItemString(PdDict, "py4pdOBJPlayable");
  PyObject *pyOUT = PyDict_GetItemString(PdDict, "py4pdOBJpyout");
  PyObject *nooutlet = PyDict_GetItemString(PdDict, "py4pdOBJnooutlet");
  PyObject *AuxOutletPy = PyDict_GetItemString(PdDict, "py4pdAuxOutlets");
  PyObject *RequireUserToSetOutletNumbers =
      PyDict_GetItemString(PdDict, "py4pdOBJrequireoutletn");
  PyObject *Py_ObjType = PyDict_GetItemString(PdDict, "py4pdOBJType");
  PyObject *pyFunction = PyDict_GetItemString(PdDict, "py4pdOBJFunction");

  int AuxOutlet = PyLong_AsLong(AuxOutletPy);
  int requireNofOutlets = PyLong_AsLong(RequireUserToSetOutletNumbers);
  PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(pyFunction);
  t_class *object_PY4PD_Class = (t_class *)PyLong_AsVoidPtr(PY_objectClass);
  t_py *x = (t_py *)pd_new(object_PY4PD_Class);

  x->visMode = 0;
  x->pyObject = 1;
  x->objArgsCount = argc;
  x->stackLimit = 100;
  x->canvas = canvas_getcurrent();
  t_canvas *c = x->canvas;
  t_symbol *patch_dir = canvas_getdir(c);
  x->objName = gensym(objectName);
  x->ignoreOnNone = PyLong_AsLong(ignoreOnNone);
  x->outPyPointer = PyLong_AsLong(pyOUT);
  x->funcCalled = 1;
  x->pFunction = pyFunction;
  x->pdPatchPath = patch_dir; // set name of the home path
  x->pkgPath = patch_dir;     // set name of the packages path
  x->playable = PyLong_AsLong(playable);
  x->pFuncName = gensym(PyUnicode_AsUTF8(code->co_name));
  x->pScriptName = gensym(PyUnicode_AsUTF8(code->co_filename));
  x->objType = PyLong_AsLong(Py_ObjType);
  Py4pdUtils_SetObjConfig(x);

  if (x->objType == PY4PD_VISOBJ)
    Py4pdUtils_CreatePicObj(x, PdDict, object_PY4PD_Class, argc, argv);

  x->pArgsCount = 0;
  int parseArgsRight = Py4pdUtils_ParseLibraryArguments(x, code, &argc, &argv);
  if (parseArgsRight == 0) {
    pd_error(NULL, "[%s] Error to parse arguments.", objectName);
    Py_DECREF(pd_module);
    Py_DECREF(py4pd_capsule);
    return NULL;
  }

  x->pdObjArgs = malloc(sizeof(t_atom) * argc);
  for (int i = 0; i < argc; i++) {
    x->pdObjArgs[i] = argv[i];
  }
  x->objArgsCount = argc;
  if (x->pyObjArgs == NULL) {
    x->pyObjArgs = malloc(sizeof(t_py4pd_pValue *) * x->pArgsCount);
  }
  int inlets = Py4pdLib_CreateObjInlets(pyFunction, x, argc, argv);
  if (inlets != 0) {
    free(x->pdObjArgs);
    free(x->pyObjArgs);
    return NULL;
  }

  if (x->objType > 1)
    x->pArgTuple = PyTuple_New(x->pArgsCount);

  if (!PyLong_AsLong(nooutlet)) {
    if (x->objType != PY4PD_AUDIOOUTOBJ && x->objType != PY4PD_AUDIOOBJ)
      x->mainOut = outlet_new(&x->obj, 0);
    else {
      x->mainOut = outlet_new(&x->obj, &s_signal);
      x->numOutlets = 1;
    }
  }
  if (requireNofOutlets) {
    if (x->numOutlets == -1) {
      pd_error(NULL,
               "[%s]: This function require that you set the number of "
               "outlets "
               "using -outn {number_of_outlets} flag",
               objectName);
      Py_DECREF(pd_module);
      Py_DECREF(py4pd_capsule);
      return NULL;
    } else
      AuxOutlet = x->numOutlets;
  }
  x->outAUX = (py4pdOuts *)getbytes(AuxOutlet * sizeof(*x->outAUX));
  x->outAUX->u_outletNumber = AuxOutlet;
  t_atom defarg[AuxOutlet], *ap;
  py4pdOuts *u;
  int i;

  if (x->objType > 1) {
    int numpyArrayImported = Py4pd_ImportNumpyForPy4pd();
    if (numpyArrayImported == 1) {
      x->numpyImported = 1;
      logpost(NULL, 3, "Numpy Loaded");
    } else {
      x->numpyImported = 0;
      pd_error(NULL, "[%s] Numpy was not imported!", objectName);
      Py_DECREF(pd_module);
      Py_DECREF(py4pd_capsule);
      return NULL;
    }
    if (x->objType == PY4PD_AUDIOOUTOBJ)
      x->audioError = 1;
  }

  for (i = 0, u = x->outAUX, ap = defarg; i < AuxOutlet; i++, u++, ap++) {
    u->u_outlet = outlet_new(&x->obj, &s_anything);
  }
  object_count++; // To clear memory when closing the patch
  Py_DECREF(pd_module);
  Py_DECREF(py4pd_capsule);
  return (x);
}

// ===========================================
// ================ PYTHON MOD ===============
// ===========================================
PyObject *Py4pdLib_AddObj(PyObject *self, PyObject *args, PyObject *keywords) {
  (void)self;
  char *objectName;
  const char *helpPatch;
  PyObject *Function;
  int w = 250, h = 250;
  int objpyout = 0;
  int nooutlet = 0;
  int added2pd_info = 0;
  int personalisedHelp = 0;
  int ignoreNoneReturn = 0;
  const char *gifImage = NULL;
  int auxOutlets = 0;
  int require_outlet_n = 0;
  int playableInt = 0;
  t_py *py4pd = Py4pdUtils_GetObject(self);

  if (py4pd->libraryFolder == NULL) {
    pd_error(py4pd, "[py4pd] Library Folder is NULL, some help patches may "
                    "not be found");
  }

  const char *helpFolder = "/help/";
  size_t totalLength =
      strlen(py4pd->libraryFolder->s_name) + strlen(helpFolder) + 1;
  char *helpFolderCHAR = (char *)malloc(totalLength * sizeof(char));
  if (helpFolderCHAR == NULL) {
    pd_error(py4pd, "[py4pd] Error allocating memory (code 001)");
    return NULL;
  }
  strcpy(helpFolderCHAR, py4pd->libraryFolder->s_name);
  strcat(helpFolderCHAR, helpFolder);

  if (!PyArg_ParseTuple(args, "Os", &Function, &objectName)) {
    post("[Python]: Error parsing arguments");
    free(helpFolderCHAR);
    return NULL;
  }
  int objectType = 0;
  if (keywords != NULL) {
    if (PyDict_Contains(keywords, PyUnicode_FromString("objtype"))) {
      PyObject *type = PyDict_GetItemString(keywords, "objtype");
      objectType = PyLong_AsLong(type);
    }
    if (PyDict_Contains(keywords, PyUnicode_FromString("figsize"))) {
      PyObject *figsize = PyDict_GetItemString(keywords, "figsize");
      PyObject *width = PyTuple_GetItem(figsize, 0);
      PyObject *height = PyTuple_GetItem(figsize, 1);
      w = PyLong_AsLong(width);
      h = PyLong_AsLong(height);
    }
    if (PyDict_Contains(keywords, PyUnicode_FromString("pyout"))) {
      PyObject *output = PyDict_GetItemString(keywords, "pyout");
      if (output == Py_True) {
        objpyout = 1;
      }
    }
    if (PyDict_Contains(keywords, PyUnicode_FromString("no_outlet"))) {
      PyObject *output = PyDict_GetItemString(keywords, "no_outlet");
      if (output == Py_True) {
        nooutlet = 1;
      }
    }
    if (PyDict_Contains(keywords, PyUnicode_FromString("require_outlet_n"))) {
      PyObject *output = PyDict_GetItemString(keywords, "require_outlet_n");
      if (output == Py_True) {
        require_outlet_n = 1;
      }
    }
    if (PyDict_Contains(keywords, PyUnicode_FromString("added2pd_info"))) {
      PyObject *output = PyDict_GetItemString(keywords, "added2pd_info");
      if (output == Py_True) {
        added2pd_info = 1;
      }
    }
    if (PyDict_Contains(keywords, PyUnicode_FromString("helppatch"))) {
      PyObject *helpname = PyDict_GetItemString(keywords, "helppatch");
      helpPatch = PyUnicode_AsUTF8(helpname);
      const char *suffix = "-help.pd";
      int suffixLen = strlen(suffix);
      int helpPatchLen = strlen(helpPatch);

      if (helpPatchLen >= suffixLen) {
        if (strcmp(helpPatch + (helpPatchLen - suffixLen), suffix) == 0) {
          personalisedHelp = 1;
          char helpPatchName[MAXPDSTRING];

          // Use snprintf to avoid buffer overflow and ensure null-termination
          snprintf(helpPatchName, sizeof(helpPatchName), "%.*s",
                   helpPatchLen - suffixLen, helpPatch);

          helpPatch = helpPatchName;
        } else {
          pd_error(NULL, "Help patch must end with '-help.pd'");
        }
      } else {
        pd_error(NULL, "Help patch must end with '-help.pd'");
      }
    }
    if (PyDict_Contains(keywords, PyUnicode_FromString("ignore_none_return"))) {
      PyObject *noneReturn =
          PyDict_GetItemString(keywords, "ignore_none_return");
      if (noneReturn == Py_True) {
        ignoreNoneReturn = 1;
      }
    }
    if (PyDict_Contains(keywords, PyUnicode_FromString("objimage"))) {
      PyObject *type = PyDict_GetItemString(keywords, "objimage");
      gifImage = PyUnicode_AsUTF8(type);
    }
    if (PyDict_Contains(keywords, PyUnicode_FromString("num_aux_outlets"))) {
      PyObject *type = PyDict_GetItemString(keywords, "num_aux_outlets");
      auxOutlets = PyLong_AsLong(type);
    }
    if (PyDict_Contains(keywords, PyUnicode_FromString("playable"))) {
      PyObject *playable = PyDict_GetItemString(keywords, "playable");
      if (playable == Py_True) {
        playableInt = 1;
      }
    }
  }
  class_set_extern_dir(gensym(helpFolderCHAR));
  t_class *localClass;

  if (objectType == PY4PD_NORMALOBJ) {
    localClass = class_new(gensym(objectName), (t_newmethod)Py4pdLib_NewObj,
                           (t_method)Py4pdLib_FreeObj, sizeof(t_py),
                           CLASS_DEFAULT, A_GIMME, 0);
  } else if (objectType == PY4PD_VISOBJ) {
    localClass = class_new(gensym(objectName), (t_newmethod)Py4pdLib_NewObj,
                           (t_method)Py4pdLib_FreeObj, sizeof(t_py),
                           CLASS_DEFAULT, A_GIMME, 0);
  } else if (objectType == PY4PD_AUDIOINOBJ) {
    localClass = class_new(gensym(objectName), (t_newmethod)Py4pdLib_NewObj,
                           (t_method)Py4pdLib_FreeObj, sizeof(t_py),
                           CLASS_MULTICHANNEL, A_GIMME, 0);
  } else if (objectType == PY4PD_AUDIOOUTOBJ) {
    localClass = class_new(gensym(objectName), (t_newmethod)Py4pdLib_NewObj,
                           (t_method)Py4pdLib_FreeObj, sizeof(t_py),
                           CLASS_MULTICHANNEL, A_GIMME, 0);
  } else if (objectType == PY4PD_AUDIOOBJ) {
    localClass = class_new(gensym(objectName), (t_newmethod)Py4pdLib_NewObj,
                           (t_method)Py4pdLib_FreeObj, sizeof(t_py),
                           CLASS_MULTICHANNEL, A_GIMME, 0);
  } else {
    PyErr_SetString(PyExc_TypeError,
                    "Object type not supported, check the spelling");
    return NULL;
  }

  // Add configs to the object
  PyObject *nestedDict = PyDict_New(); // New
  PyDict_SetItemString(nestedDict, "py4pdOBJFunction", Function);

  PyObject *Py_ObjType = PyLong_FromLong(objectType);
  PyDict_SetItemString(nestedDict, "py4pdOBJType", Py_ObjType);
  Py_DECREF(Py_ObjType);

  PyObject *Py_LibraryFolder =
      PyUnicode_FromString(py4pd->libraryFolder->s_name);
  PyDict_SetItemString(nestedDict, "py4pdOBJLibraryFolder", Py_LibraryFolder);
  Py_DECREF(Py_LibraryFolder);

  PyObject *Py_ClassLocal = PyLong_FromVoidPtr(localClass);
  PyDict_SetItemString(nestedDict, "py4pdOBJ_CLASS", Py_ClassLocal);
  Py_DECREF(Py_ClassLocal);

  PyObject *Py_Width = PyLong_FromLong(w);
  PyDict_SetItemString(nestedDict, "py4pdOBJwidth", Py_Width);
  Py_DECREF(Py_Width);

  PyObject *Py_Height = PyLong_FromLong(h);
  PyDict_SetItemString(nestedDict, "py4pdOBJheight", Py_Height);
  Py_DECREF(Py_Height);

  PyObject *Py_Playable = PyLong_FromLong(playableInt);
  PyDict_SetItemString(nestedDict, "py4pdOBJPlayable", Py_Playable);
  Py_DECREF(Py_Playable);

  if (gifImage != NULL) {
    PyObject *Py_GifImage = PyUnicode_FromString(gifImage);
    PyDict_SetItemString(nestedDict, "py4pdOBJGif", Py_GifImage);
    Py_DECREF(Py_GifImage);
  }

  PyObject *Py_ObjOuts = PyLong_FromLong(objpyout);
  PyDict_SetItemString(nestedDict, "py4pdOBJpyout", PyLong_FromLong(objpyout));
  Py_DECREF(Py_ObjOuts);

  PyObject *Py_NoOutlet = PyLong_FromLong(nooutlet);
  PyDict_SetItemString(nestedDict, "py4pdOBJnooutlet", Py_NoOutlet);
  Py_DECREF(Py_NoOutlet);

  PyObject *Py_RequireOutletN = PyLong_FromLong(require_outlet_n);
  PyDict_SetItemString(nestedDict, "py4pdOBJrequireoutletn", Py_RequireOutletN);
  Py_DECREF(Py_RequireOutletN);

  PyObject *Py_auxOutlets = PyLong_FromLong(auxOutlets);
  PyDict_SetItemString(nestedDict, "py4pdAuxOutlets", Py_auxOutlets);
  Py_DECREF(Py_auxOutlets);

  PyObject *Py_ObjName = PyUnicode_FromString(objectName);
  PyDict_SetItemString(nestedDict, "py4pdOBJname", Py_ObjName);
  Py_DECREF(Py_ObjName);

  PyObject *Py_IgnoreNoneReturn = PyLong_FromLong(ignoreNoneReturn);
  PyDict_SetItemString(nestedDict, "py4pdOBJIgnoreNone", Py_IgnoreNoneReturn);
  Py_DECREF(Py_IgnoreNoneReturn);

  PyObject *objectDict = PyDict_New();
  PyDict_SetItemString(objectDict, objectName, nestedDict);
  PyObject *py4pd_capsule = PyCapsule_New(objectDict, objectName, NULL);
  char py4pd_objectName[MAXPDSTRING];
  sprintf(py4pd_objectName, "py4pd_ObjectDict_%s", objectName);

  PyObject *pdModule = PyImport_ImportModule("pd");
  PyModule_AddObject(pdModule, py4pd_objectName, py4pd_capsule);

  Py_DECREF(pdModule);

  // =====================================
  PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(Function);
  int py_args = code->co_argcount;

  // special methods
  if (objectType == PY4PD_NORMALOBJ) {
    class_addmethod(localClass, (t_method)Py4pdLib_Click, gensym("click"), 0,
                    0);
    if (playableInt == 1) {
      class_addmethod(localClass, (t_method)Py4pdLib_Play, gensym("play"),
                      A_GIMME, 0);
      class_addmethod(localClass, (t_method)Py4pdLib_Stop, gensym("stop"), 0,
                      0);
      class_addmethod(localClass, (t_method)Py4pdLib_Clear, gensym("clear"), 0,
                      0);
    }
  } else if (objectType == PY4PD_VISOBJ) {
    if (playableInt == 1) {
      class_addmethod(localClass, (t_method)Py4pdLib_Play, gensym("play"),
                      A_GIMME, 0);
      class_addmethod(localClass, (t_method)Py4pdLib_Stop, gensym("stop"), 0,
                      0);
      class_addmethod(localClass, (t_method)Py4pdLib_Clear, gensym("clear"), 0,
                      0);
    }
    class_addmethod(localClass, (t_method)Py4pdPic_Zoom, gensym("zoom"), A_CANT,
                    0);
    class_addmethod(localClass, (t_method)Py4pd_SetPythonPointersUsage,
                    gensym("pointers"), A_FLOAT, 0);
    // class_setsavefn(localClass, &Py4pdLib_Py4pdObjPicSave);
  }
  // AUDIOIN
  else if (objectType == PY4PD_AUDIOINOBJ) {
    class_addmethod(localClass, (t_method)Py4pdLib_Click, gensym("click"), 0,
                    0);
    class_addmethod(localClass, (t_method)Py4pdLib_Dsp, gensym("dsp"), A_CANT,
                    0); // add a method to a class
    CLASS_MAINSIGNALIN(localClass, t_py, py4pdAudio);
  }
  // AUDIOOUT
  else if (objectType == PY4PD_AUDIOOUTOBJ) {
    class_addmethod(localClass, (t_method)Py4pdLib_Click, gensym("click"), 0,
                    0);
    class_addmethod(localClass, (t_method)Py4pdLib_Dsp, gensym("dsp"), A_CANT,
                    0); // add a method to a class

  }
  // AUDIO
  else if (objectType == PY4PD_AUDIOOBJ) {
    class_addmethod(localClass, (t_method)Py4pdLib_Click, gensym("click"), 0,
                    0);
    class_addmethod(localClass, (t_method)Py4pdLib_Dsp, gensym("dsp"), A_CANT,
                    0); // add a method to a class
    CLASS_MAINSIGNALIN(localClass, t_py, py4pdAudio);
  } else {
    PyErr_SetString(PyExc_TypeError,
                    "Object type not supported, check the spelling");
    return NULL;
  }
  // add methods to the class
  class_addanything(localClass, Py4pdLib_Anything);
  class_addmethod(localClass, (t_method)Py4pdLib_Pointer, gensym("PyObject"),
                  A_SYMBOL, A_POINTER, 0);
  class_addmethod(localClass, (t_method)Py4pd_PrintDocs, gensym("doc"), 0, 0);
  class_addmethod(localClass, (t_method)Py4pdLib_SetKwargs, gensym("kwargs"),
                  A_GIMME, 0);
  class_addmethod(localClass, (t_method)Py4pd_SetPythonPointersUsage,
                  gensym("pointers"), A_FLOAT, 0);
  class_addmethod(localClass, (t_method)Py4pdLib_ReloadObject, gensym("reload"),
                  0, 0);

  // add help patch
  if (personalisedHelp == 1) {
    class_sethelpsymbol(localClass, gensym(helpPatch));
  } else {
    class_sethelpsymbol(localClass, gensym(objectName));
  }
  free(helpFolderCHAR);

  if (py_args != 0) {
    py4pdInlets_proxy_class =
        class_new(gensym("_py4pdInlets_proxy"), 0, 0,
                  sizeof(t_py4pdInlet_proxy), CLASS_DEFAULT, 0);
    class_addanything(py4pdInlets_proxy_class, Py4pdLib_ProxyAnything);
    class_addmethod(py4pdInlets_proxy_class, (t_method)Py4pdLib_ProxyPointer,
                    gensym("PyObject"), A_SYMBOL, A_POINTER, 0);
  }
  if (added2pd_info == 1) {
    post("[py4pd]: Object {%s} added to PureData", objectName);
  }
  class_set_extern_dir(&s_);
  post("Object {%s} added to PureData", objectName);
  Py_RETURN_TRUE;
}
