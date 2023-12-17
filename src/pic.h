// clang-format off
#ifndef PY4PD_PIC_H
#define PY4PD_PIC_H

#include "py4pd.h"

extern void Py4pdPic_Free(t_py *x);
extern void Py4pdPic_Zoom(t_py *x, t_floatarg f);
extern void Py4pdPic_InitVisMode(t_py *x, t_canvas *c, t_symbol *py4pdArgs, int index, int argc, t_atom *argv, t_class *obj_class);
extern void Py4pdPic_ErasePic(t_py* x, struct _glist *glist); 
const  char *Py4pdPic_Filepath(t_py *x, const char *filename);
extern void Py4pdPic_Draw(t_py* x, struct _glist *glist, t_floatarg vis);
extern void Py4pdPic_GetRect(t_gobj *z, t_glist *glist, int *xp1, int *yp1, int *xp2, int *yp2);
extern void Py4pdPic_Displace(t_gobj *z, t_glist *glist, int dx, int dy);
extern void Py4pdPic_Select(t_gobj *z, t_glist *glist, int state);
extern void Py4pdPic_Delete(t_gobj *z, t_glist *glist);

// typedef struct _py4pd_edit_proxy{ 
//     t_object    p_obj;
//     t_symbol   *p_sym;
//     t_clock    *p_clock;
//     struct      _py *p_cnv;
// }t_py4pd_edit_proxy;

#endif


