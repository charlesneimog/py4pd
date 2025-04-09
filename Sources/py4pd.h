#ifndef PY4PD_H
#define PY4PD_H

#include <m_pd.h>

#include <g_canvas.h>
#include <m_imp.h>
#include <s_stuff.h>

#include <Python.h>

#define PY4PD_MAJOR_VERSION 1
#define PY4PD_MINOR_VERSION 0
#define PY4PD_MICRO_VERSION 0

static t_class *py4pd_class;
#define PYOBJECT -1997
static int objCount = 0;

// ╭─────────────────────────────────────╮
// │               Classes               │
// ╰─────────────────────────────────────╯
typedef struct _pdpy_pyclass t_pdpy_pyclass;
typedef struct _pdpy_clock t_pdpy_clock;
static t_class *pdpy_proxyinlet_class = NULL;
static t_class *pdpy_pyobjectout_class = NULL;
static t_class *pdpy_proxyclock_class = NULL;
static t_widgetbehavior pdpy_widgetbehavior;

// ╭─────────────────────────────────────╮
// │            GUI INTERFACE            │
// ╰─────────────────────────────────────╯
extern void pdpy_getrect(t_gobj *z, t_glist *glist, int *xp1, int *yp1, int *xp2, int *yp2);
extern void pdpy_displace(t_gobj *z, t_glist *glist, int dx, int dy);
extern void pdpy_delete(t_gobj *z, t_glist *glist);
extern int pdpy_click(t_gobj *z, t_glist *gl, int xpos, int ypos, int shift, int alt, int dbl,
                      int doit);
extern void pdpy_vis(t_gobj *z, t_glist *glist, int vis);
extern void pdpy_activate(t_gobj *z, t_glist *glist, int state);

// ╭─────────────────────────────────────╮
// │        External Definitions         │
// ╰─────────────────────────────────────╯
extern int sys_trytoopenone(const char *dir, const char *name, const char *ext, char *dirresult,
                            char **nameresult, unsigned int size, int bin);

#define trytoopenone(dir, name, ...)                                                               \
    sys_trytoopenone(sys_isabsolutepath(name) ? "" : dir, name, __VA_ARGS__)

// ╭─────────────────────────────────────╮
// │          Object Base Class          │
// ╰─────────────────────────────────────╯
typedef struct _pdpy_objptr {
    t_pd x_pd;
    t_symbol *id;
    PyObject *pValue;
} t_pdpy_objptr;

// ─────────────────────────────────────
typedef struct _pdpy_pdobj {
    t_object obj;
    t_sample sample;
    t_canvas *canvas;

    // PyClass
    const char *script_filename;
    PyObject *pyclass;
    char id[MAXPDSTRING];

    // dsp
    PyObject *dspfunction;
    unsigned nchs;
    unsigned vecsize;
    unsigned siginlets;
    unsigned sigoutlets;

    // gui
    int width, height;
    int mouse_drag_x, mouse_drag_y, mouse_down;
    bool has_gui;
    bool first_draw;
    int num_layers;
    char object_tag[128];

    // clock
    t_pdpy_clock **clocks;
    int clocks_size;

    // properties
    t_symbol *current_frame;
    t_symbol *properties_receiver;
    int checkbox_count;
    int current_row;
    int current_col;

    // in and outs
    t_outlet **outs;
    t_inlet **ins;
    int outletsize;
    int inletsize;
    struct pdpy_proxyinlet *proxy_in;
    t_pdpy_objptr *outobjptr; // PyObject <type> <pointer>
} t_pdpy_pdobj;

// ─────────────────────────────────────
typedef struct _pdpy_pyclass {
    PyObject_HEAD const char *name;
    t_pdpy_pdobj *pdobj;
    PyObject *outlets;
    PyObject *inlets;
    PyObject *pyargs;
} t_pdpy_pyclass;

// ─────────────────────────────────────
typedef struct pdpy_proxyinlet {
    t_pd pd;
    t_pdpy_pdobj *owner;
    unsigned int id;
} t_pdpy_proxyinlet;

// ─────────────────────────────────────
typedef struct _pdpy_clock {
    PyObject_HEAD PyObject *function;
    t_pd pd;
    t_clock *clock;
    t_pdpy_pdobj *owner;
    const char *functionname;
    float delay_time;
} t_pdpy_clock;

// ╭─────────────────────────────────────╮
// │             Definitions             │
// ╰─────────────────────────────────────╯
extern int pd4pd_loader_wrappath(int fd, const char *name, const char *dirbuf);

extern PyMODINIT_FUNC pdpy_initpuredatamodule();
extern void pdpy_proxyinlet_setup(void);
extern void pdpy_pyobjectoutput_setup(void);
extern void pdpy_execute(t_pdpy_pdobj *x, char *methodname, t_symbol *s, int argc, t_atom *argv);
extern void pdpy_proxyinlet_init(t_pdpy_proxyinlet *p, t_pdpy_pdobj *owner, unsigned int id);
extern void pdpy_proxy_anything(t_pdpy_proxyinlet *proxy, t_symbol *s, int argc, t_atom *argv);
extern void pdpy_clock_execute(t_pdpy_clock *x);
extern void pdpy_printerror(t_pdpy_pdobj *x);
extern void pdpy_pyobject(t_pdpy_pdobj *x, t_symbol *s, t_symbol *id);
extern void pdpy_execute(t_pdpy_pdobj *x, char *methodname, t_symbol *s, int argc, t_atom *argv);
extern void pdpy_properties(t_gobj *z, t_glist *owner);

#define trytoopenone(dir, name, ...)                                                               \
    sys_trytoopenone(sys_isabsolutepath(name) ? "" : dir, name, __VA_ARGS__)

extern PyObject *pdpy_getoutptr(t_symbol *s);
extern PyObject *pdpy_newclock(PyObject *self, PyObject *args);
extern PyObject *py4pdobj_converttopy(int argc, t_atom *argv);

extern PyTypeObject pdpy_type;

// ─────────────────────────────────────
typedef struct _py {
    t_object obj;
} t_py;

#endif
