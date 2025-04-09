#include "py4pd.h"

// ─────────────────────────────────────
void pdpy_proxy_anything(t_pdpy_proxyinlet *proxy, t_symbol *s, int argc, t_atom *argv) {
    t_pdpy_pdobj *o = proxy->owner;
    char methodname[MAXPDSTRING];

    if (strcmp(s->s_name, "PyObject") == 0) {
        char *str = strdup(atom_getsymbol(argv)->s_name);
        char *p = str;
        while ((p = strchr(p, '.')) != NULL) {
            *p = '_';
        }
        pdpy_pyobject(proxy->owner, gensym(str), atom_getsymbol(argv + 1));
        free(str);
    } else {
        pd_snprintf(methodname, MAXPDSTRING, "in_%d_%s", proxy->id, s->s_name);
        pdpy_execute(o, methodname, s, argc, argv);
    }

    return;
}

// ─────────────────────────────────────
static void pdpy_anything(t_pdpy_pdobj *o, t_symbol *s, int argc, t_atom *argv) {
    char methodname[MAXPDSTRING];
    pd_snprintf(methodname, MAXPDSTRING, "in_1_%s", s->s_name);
    pdpy_execute(o, methodname, s, argc, argv);
}

// ─────────────────────────────────────
void pdpy_pyobject(t_pdpy_pdobj *x, t_symbol *s, t_symbol *id) {
    if (x->pyclass == NULL) {
        post("pyclass is NULL");
        return;
    }

    char methodname[MAXPDSTRING];
    pd_snprintf(methodname, MAXPDSTRING, "in_1_pyobj_%s", s->s_name);

    PyObject *pyclass = (PyObject *)x->pyclass;
    PyObject *method = PyObject_GetAttrString(pyclass, methodname);
    if (!method || !PyCallable_Check(method)) {
        PyErr_Clear();
        pd_error(x, "[%s] method '%s' not found or not callable", x->obj.te_g.g_pd->c_name->s_name,
                 methodname);
        return;
    }

    PyObject *pArg = pdpy_getoutptr(id);
    PyObject *pValue = PyObject_CallOneArg(method, pArg);
    if (pValue == NULL) {
        pdpy_printerror(x);
        return;
    }

    if (!Py_IsNone(pValue)) {
        pd_error(x,
                 "[%s] '%s' not return None, py4pd method functions must return None, use "
                 "'self.out' to output data to "
                 "outputs.",
                 x->obj.te_g.g_pd->c_name->s_name, methodname);
    }

    Py_DECREF(method);
    Py_XDECREF(pArg);
    Py_XDECREF(pValue);

    return;
}
// ─────────────────────────────────────
static void pdpy_proxyinlet_fwd(t_pdpy_proxyinlet *p, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (!argc) {
        return;
    }

    //
    char methodname[MAXPDSTRING];
    pd_snprintf(methodname, MAXPDSTRING, "in_%d_%s", p->id, atom_getsymbol(argv)->s_name);
    pdpy_execute(p->owner, methodname, atom_getsymbol(argv), argc - 1, argv + 1);
}

// ─────────────────────────────────────
void pdpy_proxyinlet_setup(void) {
    pdpy_proxyinlet_class =
        class_new(gensym("py4pd proxy inlet"), 0, 0, sizeof(t_pdpy_proxyinlet), 0, 0);
    if (pdpy_proxyinlet_class) {
        class_addanything(pdpy_proxyinlet_class, pdpy_proxy_anything);
        class_addmethod(pdpy_proxyinlet_class, (t_method)pdpy_proxyinlet_fwd, gensym("fwd"),
                        A_GIMME, 0);
    }
}
