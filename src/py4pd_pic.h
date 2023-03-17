#include "py4pd.h"

extern t_class *py4pd_class, *py4pd_class_VIS;
extern void PY4PD_draw_io_let(t_py *x);
extern const char* PY4PD_filepath(t_py *x, const char *filename);
extern void PY4PD_mouserelease(t_py* x);
extern void PY4PD_get_snd_rcv(t_py* x);
extern int PY4PD_click(t_py *x, struct _glist *glist, int xpos, int ypos, int shift, int alt, int dbl, int doit);
extern void PY4PD_getrect(t_gobj *z, t_glist *glist, int *xp1, int *yp1, int *xp2, int *yp2);
extern void PY4PD_displace(t_gobj *z, t_glist *glist, int dx, int dy);
extern void PY4PD_select(t_gobj *z, t_glist *glist, int state);
extern void PY4PD_delete(t_gobj *z, t_glist *glist);
extern void PY4PD_draw(t_py* x, struct _glist *glist, t_floatarg vis);
extern void PY4PD_erase(t_py* x, struct _glist *glist); 
extern void PY4PD_save(t_gobj *z, t_binbuf *b);
extern void PY4PD_size_callback(t_py *x, t_float w, t_float h);
extern void PY4PD_vis(t_gobj *z, t_glist *glist, int vis); 
extern void PY4PD_open(t_py* x, t_symbol *filename); 
extern void PY4PD_send(t_py *x, t_symbol *s);
extern void PY4PD_receive(t_py *x, t_symbol *s);
extern void PY4PD_outline(t_py *x, t_floatarg f);
extern void PY4PD_size(t_py *x, t_float f);
extern void PY4PD_latch(t_py *x, t_float f);
extern void PY4PD_zoom(t_py *x, t_floatarg f);
// extern void PY4PD_properties(t_gobj *z, t_glist *gl);
extern void PY4PD_ok(t_py *x, t_symbol *s, int ac, t_atom *av);
extern void PY4PD_edit_proxy_free(t_py4pd_edit_proxy *p);
extern t_py4pd_edit_proxy * PY4PD_edit_proxy_new(t_py *x, t_symbol *s);
extern void PY4PD_free(t_py *x);
extern void py4pd_picDefintion(char *imageData);
extern void py4pd_InitVisMode(t_py *x, t_canvas *c , t_symbol *py4pdArgs, int index, int argc, t_atom *argv);

