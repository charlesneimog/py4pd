#ifndef PY4PD_PIC_H
#define PY4PD_PIC_H

#include "py4pd.h"

extern t_class *py4pd_class, *py4pd_class_VIS, *pyNewObject_VIS;
extern void PY4PD_free(t_py *x);
extern void PY4PD_zoom(t_py *x, t_floatarg f);
extern void py4pd_InitVisMode(t_py *x, t_canvas *c, t_symbol *py4pdArgs, int index, int argc, t_atom *argv, t_class *obj_class);
extern void PY4PD_erase(t_py* x, struct _glist *glist); 
extern void PY4PD_draw(t_py* x, struct _glist *glist, t_floatarg vis);
extern const char* PY4PD_filepath(t_py *x, const char *filename);

// widget
extern void PY4PD_getrect(t_gobj *z, t_glist *glist, int *xp1, int *yp1, int *xp2, int *yp2);
extern void PY4PD_displace(t_gobj *z, t_glist *glist, int dx, int dy);
extern void PY4PD_select(t_gobj *z, t_glist *glist, int state);
extern void PY4PD_delete(t_gobj *z, t_glist *glist);




#endif // PY4PD_PIC_H


