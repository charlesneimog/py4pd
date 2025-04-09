#include "py4pd.h"

// ╭─────────────────────────────────────╮
// │         PROPERTIES METHODS          │
// ╰─────────────────────────────────────╯
static PyObject *pyproperties_addcheckbox(PyObject *self, PyObject *args) {
    PyObject *name;
    PyObject *method;
    PyObject *init_value;

    if (!PyArg_ParseTuple(args, "OOO", &name, &method, &init_value)) {
        return NULL;
    }

    // Check if init_value is bool
    if (!PyBool_Check(init_value)) {
        PyErr_SetString(PyExc_TypeError, "init_value must be a boolean");
        return NULL;
    }

    // Access the class's dictionary
    PyObject *type = (PyObject *)Py_TYPE(self);
    PyObject *dict = ((PyTypeObject *)type)->tp_dict;

    if (dict == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "tp_dict is NULL");
        return NULL;
    }

    // Access _checkbox_counter
    PyObject *checkbox_counter = PyDict_GetItemString(dict, "__checkbox_counter");
    if (checkbox_counter == NULL) {
        PyErr_SetString(PyExc_KeyError, "_checkbox_counter not found in class dict");
        return NULL;
    }
    long checkbox_count = PyLong_AsLong(checkbox_counter);
    checkbox_count++;
    PyDict_SetItemString(dict, "__checkbox_counter", PyLong_FromLong(checkbox_count));

    //
    PyObject *current_frame = PyDict_GetItemString(dict, "__frameId");
    if (current_frame == NULL) {
        PyErr_SetString(PyExc_KeyError, "__frameId not found in class dict");
        return NULL;
    }
    const char *str_current = PyUnicode_AsUTF8(current_frame);

    char sanitized_frame[MAXPDSTRING];
    pd_snprintf(sanitized_frame, MAXPDSTRING, "%s", str_current);
    for (char *p = sanitized_frame; *p != '\0'; p++) {
        if (*p == '.') {
            *p = '_';
        }
    }

    // Generate unique variable name
    char check_variable[MAXPDSTRING];
    pd_snprintf(check_variable, MAXPDSTRING, "::checkbox%d_%s_state", checkbox_count,
                sanitized_frame);
    printf("sanitized_frame %s\n", check_variable);

    // Initialize the Tcl variable to 0 (unchecked)
    // int value = PyLong_AsLong(init_value);
    // pdgui_vmess(0, "ssi", "set", check_variable, value);

    // Build the pdsend command
    char pdsend[MAXPDSTRING];
    PyObject *py_receiver = PyDict_GetItemString(dict, "__receiver");
    const char *properties_receiver = PyUnicode_AsUTF8(py_receiver);
    pd_snprintf(pdsend, MAXPDSTRING, "eval pdsend [concat %s _properties checkbox %s $%s]",
                properties_receiver, method, check_variable);
    //
    // // Create the checkbox
    char checkid[MAXPDSTRING];
    pd_snprintf(checkid, MAXPDSTRING, "%s.check%d", current_frame, checkbox_count);
    pdgui_vmess(0, "ssssssss", "checkbutton", checkid, "-text", "TESTE TESTE", "-variable",
                check_variable, "-command", pdsend);

    pdgui_vmess(0, "sssisi", "grid", checkid, "-row", 0, "-column", 0, "-sticky", "we");
    // pdlua_properties_updaterow(o);

    Py_RETURN_NONE;
}

// ─────────────────────────────────────
static PyObject *pyproperties_addtextinput(PyObject *self, PyObject *args) {

    //
    Py_RETURN_NONE;
}

// ─────────────────────────────────────
static PyMethodDef pyproperties_methods[] = {
    {"addcheckbox", pyproperties_addcheckbox, METH_VARARGS, "Add new checkbox on value"},
    {"addtextinput", pyproperties_addtextinput, METH_VARARGS, "Add new text input"},
    {NULL, NULL, 0, NULL}};

// ─────────────────────────────────────
static PyTypeObject pyproperties_type = {PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pd_properties",
                                         .tp_basicsize = sizeof(PyObject),
                                         .tp_flags = Py_TPFLAGS_DEFAULT,
                                         .tp_doc = "MyClass objects",
                                         .tp_methods = pyproperties_methods,
                                         .tp_new = PyType_GenericNew};

// ─────────────────────────────────────
static void pdpy_properties_receiver(t_pdpy_pdobj *o, t_symbol *s, int argc, t_atom *argv) {
    if (argc < 2) {
        return;
    }
}

// ─────────────────────────────────────
static void pdpy_properties_createdialog(t_pdpy_pdobj *o) {
    pdgui_vmess(0, "ssss", "toplevel", o->properties_receiver->s_name, "-class", "DialogWindow");
    pdgui_vmess(0, "ssss", "wm", "title", o->properties_receiver->s_name,
                "{[mydialog] Properties}");
    pdgui_vmess(0, "sss", "wm", "group", o->properties_receiver->s_name, ".");
    pdgui_vmess(0, "sssii", "wm", "resizable", o->properties_receiver->s_name, 0, 0);

    pdgui_vmess(0, "sss", "wm", "transient", o->properties_receiver->s_name, "$::focused_window");
    pdgui_vmess(0, "ssss", o->properties_receiver->s_name, "configure", "-menu",
                "$::dialog_menubar");
    pdgui_vmess(0, "sssfsf", o->properties_receiver->s_name, "configure", "-padx", 0.0f, "-pady",
                0.0f);
}

// ─────────────────────────────────────
static void pdpy_properties_setupbuttons(t_pdpy_pdobj *o) {
    char buttonsId[MAXPDSTRING];
    pd_snprintf(buttonsId, MAXPDSTRING, ".%p.buttons", (void *)o);

    char buttonCancelId[MAXPDSTRING];
    char buttonApplyId[MAXPDSTRING];
    char buttonOkId[MAXPDSTRING];
    pd_snprintf(buttonCancelId, MAXPDSTRING, ".%p.buttons.cancel", (void *)o);
    pd_snprintf(buttonApplyId, MAXPDSTRING, ".%p.buttons.apply", (void *)o);
    pd_snprintf(buttonOkId, MAXPDSTRING, ".%p.buttons.ok", (void *)o);

    char destroyCommand[MAXPDSTRING];
    pd_snprintf(destroyCommand, MAXPDSTRING, "destroy .%p", (void *)o);

    // Criando o frame dos botões
    pdgui_vmess(0, "sssf", "frame", buttonsId, "-pady", 5.0f);
    pdgui_vmess(0, "ssss", "pack", buttonsId, "-fill", "x");

    // Cancel (Close window)
    pdgui_vmess(0, "ssssss", "button", buttonCancelId, "-text", "Cancel", "-command",
                destroyCommand);
    pdgui_vmess(0, "sssssisisi", "pack", buttonCancelId, "-side", "left", "-expand", 1, "-padx", 10,
                "-ipadx", 10);

    // Apply (send all data to pd and lua obj) for this must be necessary to save all the variables
    // used in the object in a char [128][MAXPDSTRING], I don't think that this is good, or there is
    // better solution?
    // TODO: Need to dev the apply command
    pdgui_vmess(0, "ssss", "button", buttonApplyId, "-text", "Apply");
    pdgui_vmess(0, "sssssisisi", "pack", buttonApplyId, "-side", "left", "-expand", 1, "-padx", 10,
                "-ipadx", 10);

    // Ok
    pdgui_vmess(0, "ssssss", "button", buttonOkId, "-text", "OK", "-command", destroyCommand);
    pdgui_vmess(0, "sssssisisi", "pack", buttonOkId, "-side", "left", "-expand", 1, "-padx", 10,
                "-ipadx", 10);
}

// ─────────────────────────────────────
void pdpy_properties(t_gobj *z, t_glist *owner) {
    t_pdpy_pdobj *o = (t_pdpy_pdobj *)z;
    PyObject *pyclass = (PyObject *)o->pyclass;
    PyObject *method = PyObject_GetAttrString(pyclass, "properties");
    if (!method || !PyCallable_Check(method)) {
        pd_error(o, "[%s] no properties method defined or not callable",
                 o->obj.te_g.g_pd->c_name->s_name);
        return;
    }

    char receiver[MAXPDSTRING];
    pd_snprintf(receiver, MAXPDSTRING, ".%p", o);
    o->properties_receiver = gensym(receiver);
    o->current_frame = NULL;
    pd_bind(&o->obj.ob_pd, o->properties_receiver); // new to unbind

    pdpy_properties_createdialog(o); // <-- create hidden window

    char frameId[MAXPDSTRING];
    snprintf(frameId, MAXPDSTRING, ".%p.main", (void *)o);
    pdgui_vmess(0, "sss", "wm", "deiconify", o->properties_receiver->s_name);
    pdgui_vmess(0, "sssf", "frame", frameId, "-padx", 15.0f, "-pady", 15.0f);
    pdgui_vmess(0, "sssssf", "pack", frameId, "-fill", "both", "-expand", 4.0f);
    pdgui_vmess(0, "sssfsf", "pack", frameId, "-pady", 10.f, "-padx", 10.f);

    // Create properties class
    if (PyType_Ready(&pyproperties_type) < 0) {
        PyErr_Print();
        return;
    }

    PyObject *pclass = (PyObject *)&pyproperties_type;
    PyObject *dict = ((PyTypeObject *)pclass)->tp_dict;

    if (dict == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "tp_dict is NULL after PyType_Ready");
        return;
    }

    PyDict_SetItemString(dict, "__frameId", PyUnicode_FromString(frameId));
    PyDict_SetItemString(dict, "__receiver", PyUnicode_FromString(receiver));
    PyDict_SetItemString(dict, "__checkbox_counter", PyLong_FromLong(0));
    PyDict_SetItemString(dict, "__colorpicker_counter", PyLong_FromLong(0));
    PyDict_SetItemString(dict, "__textinput_counter", PyLong_FromLong(0));

    // TODO: Add receiver on class to unbind it
    PyObject *p = PyObject_CallObject((PyObject *)&pyproperties_type, NULL);
    if (!p) {
        pdpy_printerror(o);
        return;
    }

    // TODO: Create an internal module
    PyObject *r = PyObject_CallOneArg(method, p);
    if (r == NULL) {
        pdpy_printerror(o);
        pdgui_vmess(0, "ss", "destroy", o->properties_receiver->s_name);
        return;
    }
    pdpy_properties_setupbuttons(o);

    return;
}
