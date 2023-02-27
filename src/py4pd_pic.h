#include "py4pd.h"

void py4pd_picDefintion(void);

extern t_class *py4pd_class, *edit_proxy_class;

extern t_edit_proxy *edit_proxy_new(t_py *x, t_symbol *s);
extern void edit_proxy_any(t_edit_proxy *p, t_symbol *s, int ac, t_atom *av);
extern void pic_getrect(t_gobj *z, t_glist *glist, int *xp1, int *yp1, int *xp2, int *yp2);
extern void pic_displace(t_gobj *z, t_glist *glist, int dx, int dy);
extern void pic_select(t_gobj *z, t_glist *glist, int state);
extern void pic_delete(t_gobj *z, t_glist *glist);
extern void pic_vis(t_gobj *z, t_glist *glist, int vis);
extern void pic_save(t_gobj *z, t_binbuf *b);
extern int pic_click(t_py *x, struct _glist *glist, int xpos, int ypos, int shift, int alt, int dbl, int doit);
extern void pic_size_callback(t_py *x, t_float w, t_float h);
