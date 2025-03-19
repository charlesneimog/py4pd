#include <m_pd.h>

#include <g_canvas.h>
#include <m_imp.h>
#include <s_stuff.h>

#include <Python.h>

extern PyMODINIT_FUNC PyInit_pd();
extern void pdpy_proxyinlet_setup(void);

// ─────────────────────────────────────
int sys_trytoopenone(const char *dir, const char *name, const char *ext, char *dirresult,
                     char **nameresult, unsigned int size, int bin);

// ──────────── Definitions ─────────
#define trytoopenone(dir, name, ...)                                                               \
    sys_trytoopenone(sys_isabsolutepath(name) ? "" : dir, name, __VA_ARGS__)
t_class *py4pd_class;
int objCount = 0;

// ─────────────────────────────────────
typedef struct _py {
    t_object obj;
    t_glist *glist;
    t_canvas *canvas;
    t_outlet *out1;
    t_inlet *in1;
} t_py;

// ─────────────────────────────────────
static int pd4pd_loader_wrappath(int fd, const char *name, const char *dirbuf) {
    char fullpath[1024];
    if (!dirbuf || !name) {
        pd_error(NULL, "Error: dirbuf or name is NULL\n");
        return 0;
    }
    pd_snprintf(fullpath, sizeof(fullpath), "%s/%s.pd_py", dirbuf, name);
    FILE *file = fopen(fullpath, "r");
    if (file == NULL) {
        PyErr_SetString(PyExc_ImportError, "Failed to open module file");
        return 0;
    }

    // Run the Python file
    if (PyRun_SimpleFile(file, fullpath) != 0) {
        fclose(file);
        PyErr_SetString(PyExc_ImportError, "Failed to execute module");
        return 0;
    }
    return 1;
}

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

    strncpy(filename, objectname, MAXPDSTRING);
    filename[MAXPDSTRING - 2] = 0;
    strcat(filename, "/");
    strncat(filename, classname, MAXPDSTRING - strlen(filename));
    filename[MAXPDSTRING - 1] = 0;
    if ((fd = trytoopenone(path, filename, ".pd_py", dirbuf, &ptr, MAXPDSTRING, 1)) >= 0)
        if (pd4pd_loader_wrappath(fd, objectname, dirbuf)) {
            return 1;
        }
    return 0;
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
        post("[py4pd] by Charles K. Neimog");
        post("[py4pd] Python version %d.%d.%d", PY_MAJOR_VERSION, PY_MINOR_VERSION,
             PY_MICRO_VERSION);
        post("");
        PyImport_AppendInittab("puredata", PyInit_pd);
        Py_Initialize();
    }

    pdpy_proxyinlet_setup();
}

#ifdef __WIN64
__declspec(dllexport) void py4pd_setup(void);
#endif
