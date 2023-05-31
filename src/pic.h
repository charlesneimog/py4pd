#ifndef PY4PD_PIC_H
#define PY4PD_PIC_H

#include "py4pd.h"

extern t_class *py4pd_class, *py4pd_class_VIS, *pyNewObject_VIS;
extern void PY4PD_free(t_py *x);
extern void PY4PD_zoom(t_py *x, t_floatarg f);
extern void py4pd_InitVisMode(t_py *x, t_canvas *c , t_symbol *py4pdArgs, int index, int argc, t_atom *argv);
extern void PY4PD_erase(t_py* x, struct _glist *glist); 
extern void PY4PD_draw(t_py* x, struct _glist *glist, t_floatarg vis);
extern const char* PY4PD_filepath(t_py *x, const char *filename);

#endif // PY4PD_PIC_H


