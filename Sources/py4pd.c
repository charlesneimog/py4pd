#include "py4pd.h"

// ╭─────────────────────────────────────╮
// │             Definitions             │
// ╰─────────────────────────────────────╯
//

// ─────────────────────────────────────
static int pd4pd_loader_pathwise(t_canvas *canvas, const char *objectname, const char *path) {
    char dirbuf[MAXPDSTRING], filename[MAXPDSTRING];
    char *ptr;
    const char *classname;
    int fd;
    if (!path) {
        return 0;
    }

    if ((classname = strrchr(objectname, '/'))) {
        classname++;
    } else {
        classname = objectname;
    }
    if ((fd = trytoopenone(path, objectname, ".pd_py", dirbuf, &ptr, MAXPDSTRING, 1)) >= 0)
        if (pd4pd_loader_wrappath(fd, objectname, dirbuf)) {
            return 1;
        }

    pd_snprintf(filename, MAXPDSTRING, "%s", objectname);
    pd_snprintf(filename + strlen(filename), MAXPDSTRING - strlen(filename), "/");
    pd_snprintf(filename + strlen(filename), MAXPDSTRING - strlen(filename), "%s", classname);
    filename[MAXPDSTRING - 1] = 0;
    if ((fd = trytoopenone(path, filename, ".pd_py", dirbuf, &ptr, MAXPDSTRING, 1)) >= 0)
        if (pd4pd_loader_wrappath(fd, objectname, dirbuf)) {
            return 1;
        }
    return 0;
}

// ─────────────────────────────────────
void py4pd_addpath2syspath(const char *path) {
    if (!path) {
        pd_error(NULL, "Invalid path: NULL");
        return;
    }

    PyObject *sys = PyImport_ImportModule("sys");
    if (!sys) {
        pd_error(NULL, "Failed to import sys module");
        return;
    }

    PyObject *sysPath = PyObject_GetAttrString(sys, "path");
    if (!sysPath || !PyList_Check(sysPath)) {
        pd_error(NULL, "Failed to access sys.path or it is not a list");
        Py_XDECREF(sysPath);
        Py_DECREF(sys);
        return;
    }

    PyObject *pathEntry = PyUnicode_FromString(path);
    if (!pathEntry) {
        pd_error(NULL, "Failed to create Python string from path");
        Py_DECREF(sysPath);
        Py_DECREF(sys);
        return;
    }

    if (PySequence_Contains(sysPath, pathEntry) == 0) {
        if (PyList_Append(sysPath, pathEntry) != 0) {
            pd_error(NULL, "Failed to append path to sys.path");
        }
    }

    Py_DECREF(pathEntry);
    Py_DECREF(sysPath);
    Py_DECREF(sys);
}

// ─────────────────────────────────────
void py4pd_set_py4pdpath_env(const char *path) {
    if (!path) {
        pd_error(NULL, "Invalid path: NULL");
        return;
    }

    PyObject *os = PyImport_ImportModule("os");
    if (!os) {
        pd_error(NULL, "Failed to import os module");
        return;
    }

    PyObject *environ = PyObject_GetAttrString(os, "environ");
    if (!environ) {
        pd_error(NULL, "Failed to access os.environ");
        Py_DECREF(os);
        return;
    }

    PyObject *key = PyUnicode_FromString("PY4PD_PATH");
    PyObject *value = PyUnicode_FromString(path);

    if (!key || !value) {
        pd_error(NULL, "Failed to create Python strings for environment key/value");
        Py_XDECREF(key);
        Py_XDECREF(value);
        Py_DECREF(environ);
        Py_DECREF(os);
        return;
    }

    if (PyObject_SetItem(environ, key, value) != 0) {
        pd_error(NULL, "Failed to set PY4PD_PATH in os.environ");
    }

    Py_DECREF(key);
    Py_DECREF(value);
    Py_DECREF(environ);
    Py_DECREF(os);
}

// ─────────────────────────────────────
void *py4pd_new(t_symbol *s, int argc, t_atom *argv) {
    t_py *x = (t_py *)pd_new(py4pd_class);
    return (void *)x;
}

// ─────────────────────────────────────
void py4pd_setup(void) {
    int Major, Minor, Micro;
    sys_getversion(&Major, &Minor, &Micro);
    sys_register_loader((loader_t)pd4pd_loader_pathwise);

    if (Major < 0 && Minor < 54) {
        pd_error(NULL, "[py4pd] py4pd requires Pd version 0.54 or later.");
        return;
    }

    py4pd_class =
        class_new(gensym("py4pd"), (t_newmethod)py4pd_new, NULL, sizeof(t_py), 0, A_GIMME, 0);

    if (!Py_IsInitialized()) {
        objCount = 0;
        post("");
        post("[py4pd] by Charles K. Neimog | version %d.%d.%d", PY4PD_MAJOR_VERSION,
             PY4PD_MINOR_VERSION, PY4PD_MICRO_VERSION);
        post("[py4pd] Python version %d.%d.%d", PY_MAJOR_VERSION, PY_MINOR_VERSION,
             PY_MICRO_VERSION);
        post("");
        int r = PyImport_AppendInittab("puredata", pdpy_initpuredatamodule);
        if (r < 0) {
            pd_error(NULL, "[py4pd] PyInit_pd failed");
            return;
        }
        Py_Initialize();

        const char *py4pd_path = py4pd_class->c_externdir->s_name;
        py4pd_set_py4pdpath_env(py4pd_path);
        char py4pd_env[MAXPDSTRING];
        pd_snprintf(py4pd_env, MAXPDSTRING, "%s/py4pd-env", py4pd_path);
        py4pd_addpath2syspath(py4pd_env);
    }

    pdpy_proxyinlet_setup();
    pdpy_pyobjectoutput_setup();
}

#ifdef __WIN64
__declspec(dllexport) void py4pd_setup(void);
#endif
