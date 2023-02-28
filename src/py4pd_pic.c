#include "py4pd_pic.h"


// ------------------------ draw inlet --------------------------------------------------------------------
static void pic_draw_io_let(t_py *x){
    t_canvas *cv = glist_getcanvas(x->x_glist);
    int xpos = text_xpix(&x->x_obj, x->x_glist), ypos = text_ypix(&x->x_obj, x->x_glist);
    sys_vgui(".x%lx.c delete %lx_in\n", cv, x);
    if(x->x_edit && x->x_receive == &s_)
        sys_vgui(".x%lx.c create rectangle %d %d %d %d -fill black -tags %lx_in\n",
            cv, xpos, ypos, xpos+(IOWIDTH*x->x_zoom), ypos+(IHEIGHT*x->x_zoom), x);
    sys_vgui(".x%lx.c delete %lx_out\n", cv, x);
    if(x->x_edit && x->x_send == &s_)
        sys_vgui(".x%lx.c create rectangle %d %d %d %d -fill black -tags %lx_out\n",
            cv, xpos, ypos+x->x_height, xpos+IOWIDTH*x->x_zoom, ypos+x->x_height-IHEIGHT*x->x_zoom, x);
}

// --------------------------------------------------------------------------------------
// helper functions
static const char* pic_filepath(t_py *x, const char *filename){
    static char fn[MAXPDSTRING];
    char *bufptr;
    int fd = canvas_open(glist_getcanvas(x->x_glist),
        filename, "", fn, &bufptr, MAXPDSTRING, 1);
    if(fd > 0){
        fn[strlen(fn)]='/';
        sys_close(fd);
        return(fn);
    }
    else
        return(0);
}

static void pic_mouserelease(t_py* x){
    if(x->x_latch){
        outlet_float(x->x_outlet, 0);
        if(x->x_send != &s_ && x->x_send->s_thing)
            pd_float(x->x_send->s_thing, 0);
    }
}

static void pic_get_snd_rcv(t_py* x){
    t_binbuf *bb = x->x_obj.te_binbuf;
    int n_args = binbuf_getnatom(bb), i = 0; // number of arguments
    char buf[128];
    if(!x->x_snd_set){ // no send set, search arguments/flags
        if(n_args > 0){ // we have arguments, let's search them
            if(x->x_flag){ // arguments are flags actually
                if(x->x_s_flag){ // we got a search flag, let's get it
                    for(i = 0;  i < n_args; i++){
                        atom_string(binbuf_getvec(bb) + i, buf, 80);
                        if(gensym(buf) == gensym("-send")){
                            i++;
                            atom_string(binbuf_getvec(bb) + i, buf, 80);
                            x->x_snd_raw = gensym(buf);
                            break;
                        }
                    }
                }
            }
            else{ // we got no flags, let's search for argument
                int arg_n = 3; // receive argument number
                if(n_args >= arg_n){ // we have it, get it
                    atom_string(binbuf_getvec(bb) + arg_n, buf, 80);
                    x->x_snd_raw = gensym(buf);
                }
            }
        }
    }
    if(x->x_snd_raw == &s_)
        x->x_snd_raw = gensym("empty");
    if(!x->x_rcv_set){ // no receive set, search arguments
        if(n_args > 0){ // we have arguments, let's search them
            if(x->x_flag){ // arguments are flags actually
                if(x->x_r_flag){ // we got a receive flag, let's get it
                    for(i = 0;  i < n_args; i++){
                        atom_string(binbuf_getvec(bb) + i, buf, 80);
                        if(gensym(buf) == gensym("-receive")){
                            i++;
                            atom_string(binbuf_getvec(bb) + i, buf, 80);
                            x->x_rcv_raw = gensym(buf);
                            break;
                        }
                    }
                }
            }
            else{ // we got no flags, let's search for argument
                int arg_n = 4; // receive argument number
                if(n_args >= arg_n){ // we have it, get it
                    atom_string(binbuf_getvec(bb) + arg_n, buf, 80);
                    x->x_rcv_raw = gensym(buf);
                }
            }
        }
    }
    if(x->x_rcv_raw == &s_)
        x->x_rcv_raw = gensym("empty");
}

// ------------------------ pic widgetbehaviour-------------------------------------------------------------------
int pic_click(t_py *x, struct _glist *glist, int xpos, int ypos, int shift, int alt, int dbl, int doit){
    glist = NULL, xpos = ypos = shift = alt = dbl = 0;
    if(doit){
        x->x_latch ? outlet_float(x->x_outlet, 1) : outlet_bang(x->x_outlet) ;
        if(x->x_send != &s_ && x->x_send->s_thing)
            x->x_latch ? pd_float(x->x_send->s_thing, 1) : pd_bang(x->x_send->s_thing);
    }
    return(1);
}

void pic_getrect(t_gobj *z, t_glist *glist, int *xp1, int *yp1, int *xp2, int *yp2){
    t_py* x = (t_py*)z;
    int xpos = *xp1 = text_xpix(&x->x_obj, glist), ypos = *yp1 = text_ypix(&x->x_obj, glist);
    *xp2 = xpos + x->x_width, *yp2 = ypos + x->x_height;
}

void pic_displace(t_gobj *z, t_glist *glist, int dx, int dy){
    t_py *obj = (t_py *)z;
    obj->x_obj.te_xpix += dx, obj->x_obj.te_ypix += dy;
    t_canvas *cv = glist_getcanvas(glist);
    sys_vgui(".x%lx.c move %lx_outline %d %d\n", cv, obj, dx*obj->x_zoom, dy*obj->x_zoom);
    sys_vgui(".x%lx.c move %lx_picture %d %d\n", cv, obj, dx*obj->x_zoom, dy*obj->x_zoom);
    if(obj->x_receive == &s_)
        sys_vgui(".x%lx.c move %lx_in %d %d\n", cv, obj, dx*obj->x_zoom, dy*obj->x_zoom);
    if(obj->x_send == &s_)
        sys_vgui(".x%lx.c move %lx_out %d %d\n", cv, obj, dx*obj->x_zoom, dy*obj->x_zoom);
    canvas_fixlinesfor(glist, (t_text*)obj);
}

void pic_select(t_gobj *z, t_glist *glist, int state){
    t_py *x = (t_py *)z;
    int xpos = text_xpix(&x->x_obj, glist);
    int ypos = text_ypix(&x->x_obj, glist);
    t_canvas *cv = glist_getcanvas(glist);
    x->x_sel = state;
    if(state){
        sys_vgui(".x%lx.c delete %lx_outline\n", cv, x);
        sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline blue -width %d\n",
            cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
     }
    else{
        sys_vgui(".x%lx.c delete %lx_outline\n", cv, x);
        if(x->x_edit || x->x_outline)
            sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d\n",
                cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
    }
}

void pic_delete(t_gobj *z, t_glist *glist){
    canvas_deletelinesfor(glist, (t_text *)z);
}

static void pic_draw(t_py* x, struct _glist *glist, t_floatarg vis){
    t_canvas *cv = glist_getcanvas(glist);
    int xpos = text_xpix(&x->x_obj, x->x_glist), ypos = text_ypix(&x->x_obj, x->x_glist);
    int visible = (glist_isvisible(x->x_glist) && gobj_shouldvis((t_gobj *)x, x->x_glist));
    if(x->x_def_img && (visible || vis)){ // DEFAULT PIC
        sys_vgui(".x%lx.c create image %d %d -anchor nw -tags %lx_picture\n",
            cv, xpos, ypos, x);
        sys_vgui(".x%lx.c itemconfigure %lx_picture -image %s\n",
            cv, x, "pic_def_img");
        if(x->x_edit || x->x_outline)
            sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d\n",
                cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
    }
    else{
        if(visible || vis){
            sys_vgui("if { [info exists %lx_picname] == 1 } { .x%lx.c create image %d %d -anchor nw -image %lx_picname -tags %lx_picture\n} \n",
                x->x_fullname, cv, xpos, ypos, x->x_fullname, x);
        }
        if(!x->x_init)
            x->x_init = 1;
        else if((visible || vis) && (x->x_edit || x->x_outline))
            sys_vgui("if { [info exists %lx_picname] == 1 } {.x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d}\n",
                x->x_fullname, cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
        sys_vgui("if { [info exists %lx_picname] == 1 } {pdsend \"%s _picsize [image width %lx_picname] [image height %lx_picname]\"}\n",
             x->x_fullname, x->x_x->s_name, x->x_fullname, x->x_fullname);
    }
    sys_vgui(".x%lx.c bind %lx_picture <ButtonRelease> {pdsend [concat %s _mouserelease \\;]}\n", cv, x, x->x_x->s_name);
    pic_draw_io_let(x);
         
}


static void pic_erase(t_py* x, struct _glist *glist){
    t_canvas *cv = glist_getcanvas(glist);
    sys_vgui(".x%lx.c delete %lx_picture\n", cv, x); // ERASE
    sys_vgui(".x%lx.c delete %lx_in\n", cv, x);
    sys_vgui(".x%lx.c delete %lx_out\n", cv, x);
    sys_vgui(".x%lx.c delete %lx_outline\n", cv, x); // if edit?
}

void pic_save(t_gobj *z, t_binbuf *b){
    t_py *x = (t_py *)z;
    if(x->x_filename == &s_)
        x->x_filename = gensym("empty");
    pic_get_snd_rcv(x);
    binbuf_addv(b, "ssiisisssii", gensym("#X"), gensym("obj"), x->x_obj.te_xpix, x->x_obj.te_ypix,
        atom_getsymbol(binbuf_getvec(x->x_obj.te_binbuf)), x->x_outline, x->x_filename, x->x_snd_raw,
        x->x_rcv_raw, x->x_size, x->x_latch);
    binbuf_addv(b, ";");
}


//------------------------------- METHODS --------------------------------------------
void pic_size_callback(t_py *x, t_float w, t_float h){ // callback
    x->x_width = w;
    x->x_height = h;
    if(glist_isvisible(x->x_glist) && gobj_shouldvis((t_gobj *)x, x->x_glist)){
        t_canvas *cv = glist_getcanvas(x->x_glist);
        int xpos = text_xpix(&x->x_obj, x->x_glist), ypos = text_ypix(&x->x_obj, x->x_glist);
        sys_vgui("if { [info exists %lx_picname] == 1 } { .x%lx.c create image %d %d -anchor nw -image %lx_picname -tags %lx_picture\n} \n",
            x->x_fullname, cv, xpos, ypos, x->x_fullname, x);
        canvas_fixlinesfor(x->x_glist, (t_text*)x);
        if(x->x_edit || x->x_outline){
            sys_vgui(".x%lx.c delete %lx_outline\n", cv, x);
            if(x->x_sel)
                sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline blue -width %d\n",
                cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
            else
                sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d\n",
                cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
            pic_draw_io_let(x);
        }
    }
    else
        pic_erase(x, x->x_glist);
    if(x->x_size){
        t_atom at[2];
        SETFLOAT(at, w);
        SETFLOAT(at+1, h);
        outlet_list(x->x_obj.ob_outlet, &s_list, 2, at);
    }
}

void pic_vis(t_gobj *z, t_glist *glist, int vis){
    t_py* x = (t_py*)z;
    vis ? pic_draw(x, glist, 1) : pic_erase(x, glist);
}


void pic_open(t_py* x, t_symbol *filename){
    if(filename){
        if(filename == gensym("empty") && x->x_def_img)
            return;
        if(filename != x->x_filename){
            const char *file_name_open = pic_filepath(x, filename->s_name); // path
            if(file_name_open){
                x->x_filename = filename;
                x->x_fullname = gensym(file_name_open);
                if(x->x_def_img)
                    x->x_def_img = 0;
                if(glist_isvisible(x->x_glist) && gobj_shouldvis((t_gobj *)x, x->x_glist)){
                    pic_erase(x, x->x_glist);
                    sys_vgui("if {[info exists %lx_picname] == 0} {image create photo %lx_picname -file \"%s\"\n set %lx_picname 1\n}\n",
                        x->x_fullname, x->x_fullname, file_name_open, x->x_fullname);
                    pic_draw(x, x->x_glist, 0);
                }
            }
            else
                pd_error(x, "[pic]: error opening file '%s'", filename->s_name);
        }
    }
    else
        pd_error(x, "[pic]: open needs a file name");
}

static void pic_send(t_py *x, t_symbol *s){
    if(s != gensym("")){
        t_symbol *snd = (s == gensym("empty")) ? &s_ : canvas_realizedollar(x->x_glist, s);
        if(snd != x->x_send){
            x->x_snd_raw = s;
            x->x_send = snd;
            x->x_snd_set = 1;
            if(x->x_edit && glist_isvisible(x->x_glist) && gobj_shouldvis((t_gobj *)x, x->x_glist)){
                if(x->x_send == &s_)
                    pic_draw_io_let(x);
                else
                    sys_vgui(".x%lx.c delete %lx_out\n", glist_getcanvas(x->x_glist), x);
            }
        }
    }
}

static void pic_receive(t_py *x, t_symbol *s){
    if(s != gensym("")){
        t_symbol *rcv = s == gensym("empty") ? &s_ : canvas_realizedollar(x->x_glist, s);
        if(rcv != x->x_receive){
            if(x->x_receive != &s_)
                pd_unbind(&x->x_obj.ob_pd, x->x_receive);
            x->x_rcv_set = 1;
            x->x_rcv_raw = s;
            x->x_receive = rcv;
            if(x->x_receive == &s_){
                if(x->x_edit && glist_isvisible(x->x_glist) && gobj_shouldvis((t_gobj *)x, x->x_glist))
                    pic_draw_io_let(x);
            }
            else{
                pd_bind(&x->x_obj.ob_pd, x->x_receive);
                if(x->x_edit && glist_isvisible(x->x_glist) && gobj_shouldvis((t_gobj *)x, x->x_glist))
                    sys_vgui(".x%lx.c delete %lx_in\n", glist_getcanvas(x->x_glist), x);
            }
        }
    }
}

static void pic_outline(t_py *x, t_float f){
    int outline = (int)(f != 0);
    if(x->x_outline != outline){
        x->x_outline = outline;
        if(glist_isvisible(x->x_glist) && gobj_shouldvis((t_gobj *)x, x->x_glist)){
            t_canvas *cv = glist_getcanvas(x->x_glist);
            if(x->x_outline){
                int xpos = text_xpix(&x->x_obj, x->x_glist), ypos = text_ypix(&x->x_obj, x->x_glist);
                if(x->x_sel) sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline blue -width %d\n",
                    cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
                else sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d\n",
                    cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
            }
            else if(!x->x_edit)
                sys_vgui(".x%lx.c delete %lx_outline\n", cv, x);

        }
    }
}

static void pic_size(t_py *x, t_float f){
    int size = (int)(f != 0);
    if(x->x_size != size)
        x->x_size = size;
}

static void pic_latch(t_py *x, t_float f){
    int latch = (int)(f != 0);
    if(x->x_latch != latch){
        x->x_latch = latch;
    }
}

void edit_proxy_any(t_edit_proxy *p, t_symbol *s, int ac, t_atom *av){
    int edit = ac = 0;
    if(p->p_cnv){
        if(s == gensym("editmode"))
            edit = (int)(av->a_w.w_float);
        else if(s == gensym("obj") || s == gensym("msg") || s == gensym("floatatom")
        || s == gensym("symbolatom") || s == gensym("text") || s == gensym("bng")
        || s == gensym("toggle") || s == gensym("numbox") || s == gensym("vslider")
        || s == gensym("hslider") || s == gensym("vradio") || s == gensym("hradio")
        || s == gensym("vumeter") || s == gensym("mycnv") || s == gensym("selectall")){
            edit = 1;
        }
        else
            return;
        if(p->p_cnv->x_edit != edit){
            p->p_cnv->x_edit = edit;
            t_canvas *cv = glist_getcanvas(p->p_cnv->x_glist);
            if(edit){
                int x = text_xpix(&p->p_cnv->x_obj, p->p_cnv->x_glist);
                int y = text_ypix(&p->p_cnv->x_obj, p->p_cnv->x_glist);
                int w = p->p_cnv->x_width, h = p->p_cnv->x_height;
                if(!p->p_cnv->x_outline)
                    sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d\n",
                             cv, x, y, x+w, y+h, p->p_cnv, p->p_cnv->x_zoom);
                pic_draw_io_let(p->p_cnv);
            }
            else{
                if(!p->p_cnv->x_outline)
                    sys_vgui(".x%lx.c delete %lx_outline\n", cv, p->p_cnv);
                sys_vgui(".x%lx.c delete %lx_in\n", cv, p->p_cnv);
                sys_vgui(".x%lx.c delete %lx_out\n", cv, p->p_cnv);
            }
        }
    }
}

static void pic_zoom(t_py *x, t_floatarg zoom){
    x->x_zoom = (int)zoom;
}

//------------------- Properties --------------------------------------------------------
void pic_properties(t_gobj *z, t_glist *gl){
    gl = NULL;
    t_py *x = (t_py *)z;
    if(x->x_filename ==  &s_)
        x->x_filename = gensym("empty");
    pic_get_snd_rcv(x);
    char buffer[512];
    sprintf(buffer, "pic_properties %%s {%s} %d %d %d {%s} {%s} \n",
        x->x_filename->s_name,
        x->x_outline,
        x->x_size,
        x->x_latch,
        x->x_snd_raw->s_name,
        x->x_rcv_raw->s_name);
    gfxstub_new(&x->x_obj.ob_pd, x, buffer);
}

static void pic_ok(t_py *x, t_symbol *s, int ac, t_atom *av){
    s = NULL;
    t_atom undo[6];
    SETSYMBOL(undo+0, x->x_filename);
    SETFLOAT(undo+1, x->x_outline);
    SETFLOAT(undo+2, x->x_size);
    SETFLOAT(undo+3, x->x_latch);
    SETSYMBOL(undo+4, x->x_snd_raw);
    SETSYMBOL(undo+5, x->x_rcv_raw);
    pd_undo_set_objectstate(x->x_glist, (t_pd*)x, gensym("ok"), 6, undo, ac, av);
    pic_open(x, atom_getsymbolarg(0, ac, av));
    pic_outline(x, atom_getfloatarg(1, ac, av));
    pic_size(x, atom_getfloatarg(2, ac, av));
    pic_latch(x, atom_getfloatarg(3, ac, av));
    pic_send(x, atom_getsymbolarg(4, ac, av));
    pic_receive(x, atom_getsymbolarg(5, ac, av));
    canvas_dirty(x->x_glist, 1);
}

//-------------------------------------------------------------------------------------
static void edit_proxy_free(t_edit_proxy *p){
    pd_unbind(&p->p_obj.ob_pd, p->p_sym);
    clock_free(p->p_clock);
    pd_free(&p->p_obj.ob_pd);
}

t_edit_proxy * edit_proxy_new(t_py *x, t_symbol *s){
    t_edit_proxy *p = (t_edit_proxy*)pd_new(edit_proxy_class);
    p->p_cnv = x;
    pd_bind(&p->p_obj.ob_pd, p->p_sym = s);
    p->p_clock = clock_new(p, (t_method)edit_proxy_free);
    return(p);
}

void pic_free(t_py *x){ // delete if variable is unset and image is unused
    sys_vgui("if { [info exists %lx_picname] == 1 && [image inuse %lx_picname] == 0} { image delete %lx_picname \n unset %lx_picname\n}\n",
        x->x_fullname, x->x_fullname, x->x_fullname, x->x_fullname);
    if(x->x_receive != &s_)
        pd_unbind(&x->x_obj.ob_pd, x->x_receive);
    pd_unbind(&x->x_obj.ob_pd, x->x_x);
    x->x_proxy->p_cnv = NULL;
    clock_delay(x->x_proxy->p_clock, 0);
    gfxstub_deleteforkey(x);
}



// =====================================

void py4pd_picDefintion(void){
    sys_vgui("image create photo pic_def_img -data {R0lGODlhfgDEAPflAAAAAAEBAQICAgMDAwQEBAUFBQYGBgcHBwgICAkJCQoKCgsLCwwMDA0NDQ4ODg8PDxAQEBERERISEhMTExQUFBUVFRYWFhcXFxgYGBkZGRoaGhsbGxwcHB0dHR4eHh8fHyAgICEhISIiIiQkJCUlJSYmJicnJygoKCkpKSoqKiwsLC0tLS4uLi8vLzAwMDExMTIyMjMzMzQ0NDU1NTY2Njg4ODk5OTo6Ojs7Ozw8PD09PT4+PkBAQEFBQUJCQkNDQ0REREVFRUZGRklJSUpKSktLS0xMTE9PT1BQUFJSUlNTU1RUVFdXV1hYWFlZWVpaWltbW1xcXF1dXV5eXl9fX2BgYGFhYWJiYmNjY2RkZGVlZWZmZmdnZ2hoaGlpaWpqamtra2xsbG5ubm9vb3BwcHFxcXJycnNzc3R0dHV1dXZ2dnd3d3h4eHl5eXp6ent7e319fX5+fn9/f4CAgIGBgYKCgoODg4SEhIaGhoiIiImJiYqKiouLi4yMjI2NjY+Pj5CQkJGRkZKSkpOTk5SUlJWVlZaWlpiYmJmZmZqampubm5ycnJ2dnZ6enp+fn6CgoKGhoaKioqOjo6SkpKWlpaenp6ioqKmpqaqqqqysrK2tra6urq+vr7CwsLGxsbKysrOzs7S0tLW1tba2tre3t7i4uLm5ubq6uru7u7y8vL29vcHBwcLCwsTExMXFxcbGxsfHx8jIyMnJycrKysvLy8zMzM3NzdDQ0NHR0dLS0tPT09TU1NXV1dbW1tfX19jY2NnZ2dra2tvb29zc3N3d3d7e3t/f3+Dg4OHh4eLi4uPj4+Tk5Obm5ufn5+jo6Onp6erq6uvr6+zs7O3t7e7u7u/v7/Dw8PHx8fLy8vPz8/T09PX19fb29vf39/j4+Pn5+fr6+vv7+/z8/P39/f7+/v///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAAAAAAAIf8LWE1QIERhdGFYTVA8P3hwYWNrZXQgYmVnaW49J++7vycgaWQ9J1c1TTBNcENlaGlIenJlU3pOVGN6a2M5ZCc/Pgo8eDp4bXBtZXRhIHhtbG5zOng9J2Fkb2JlOm5zOm1ldGEvJyB4OnhtcHRrPSdJbWFnZTo6RXhpZlRvb2wgMTIuNDAnPgo8cmRmOlJERiB4bWxuczpyZGY9J2h0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMnPgoKIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PScnCiAgeG1sbnM6dGlmZj0naHR0cDovL25zLmFkb2JlLmNvbS90aWZmLzEuMC8nPgogIDx0aWZmOk9yaWVudGF0aW9uPjE8L3RpZmY6T3JpZW50YXRpb24+CiA8L3JkZjpEZXNjcmlwdGlvbj4KPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKPD94cGFja2V0IGVuZD0ndyc/PgH//v38+/r5+Pf29fTz8vHw7+7t7Ovq6ejn5uXk4+Lh4N/e3dzb2tnY19bV1NPS0dDPzs3My8rJyMfGxcTDwsHAv769vLu6ubi3trW0s7KxsK+urayrqqmop6alpKOioaCfnp2cm5qZmJeWlZSTkpGQj46NjIuKiYiHhoWEg4KBgH9+fXx7enl4d3Z1dHNycXBvbm1sa2ppaGdmZWRjYmFgX15dXFtaWVhXVlVUU1JRUE9OTUxLSklIR0ZFRENCQUA/Pj08Ozo5ODc2NTQzMjEwLy4tLCsqKSgnJiUkIyIhIB8eHRwbGhkYFxYVFBMSERAPDg0MCwoJCAcGBQQDAgEAACwAAAAAfgDEAAAI/gDLCRxIsKDBgwgTKhwYjpvAcOUgQoxI0SHDhRgzatzIsWNBixcRTixncaTHkyhTqhQ5MZzJhg9XypxJ8yC3cNckutzZ8lrMmkCDnsQ58Nq0a86SJp2WVBhIhyaFSp1qUOI0ODlc4Ni6NcdWJsJCUh1L1uVALQcCqF3LFhZJsnDjUmzCtm4ABLjeRowqt69KiXTtqk1ryiI3kH4Tp5wYWLDaRHwVS0YZrrFjLT8na+Y4kYngtGpdQDS8uXTGyo4DpJUQ9iZF07CrCrT8OVHs2yIjerYLWu2RaRJx4x65O3WAC8McIhZumrFxtQ3gTGNOPVxxwRLUlhjmsmRm6n2J/hsHfQAO4pfg/XZ+rhoBBlXpYQMef1ctE2fL4yd2brzB4At+7KWfZqgZd8Fxqq0Ay03oDViWQNfZhYBdORxmloAOjjUfe3VNARxEPkWWIU0j0cahBGsI5AxJIo44k0QRchhAB4ZM95qLY8XInn8SGBJRiC3hSOJsMrIFWgmQNTRSi0Ju1JKO7IEWQh4PXdikTOsVuZZ/aV3QiDM8XVkTlOxNiMABCBiy041iDhWRiVoOFoAdPhkWnED5tSlbOXDGqVYITay4ppIEMannm35+FkAIQ/TyEEhWHppbOWTGOaEEOLhVjk8M5SlpoX0+19taEwbQQAjwsfjUpwfxlyhv/tBdYImdeLIqm3V+jirYmQckMoxANhoqpniv2uUfqQHA8atZwg5LZLEcXhEWRTpF5NqnuEIragNK2OJQiNfW2iax2tK3Qi9E2XRoluWmtloJq+jFYrMObthuahhoZ8o1UIk17rP3CnbgAWlhMAlUpGEbarvZqXUpBo0swyKbzlIaMIddQjKopPbu2puuBDuGwLG8HXABJK36O2CJz7ngVQgdTKirnKqROvN/n5DU78T03gajcSHAktQxwwwjSyJwMLHCBbzeVSoCMgt27Amf1OmpkJUG8GtME3HjDCx+KBHCsSMPdnN7B4SQ12F6urrrdKPtZaXXlkxxoIz+NTCD/qOZXV3dQ1kPM83OeO7EDVLlDANHCTTXldaEXCohsYCR1guhcQhIXBKDkUEkTB4rqJbWqJfWh8AZx1QrZqUIHCNvoQwFWQ4sdkAdMqwBSIAAJ4JOxKmLBTrWwLQa6XSYKkGg2Z7DbPmHgSwRTReu5XxiHlbPPIXjzDVw1Fdz1DVzOw3cFNe78PAFYb8pN5xI8DhbpY4eYPkrX54a+j1/ZCMsJwz89McYENywWEe8xZBEGC4YHXnWkpYrXCNEI7oQAenHEZ9cQxg/aFzNRGeK/MWGXVIrIAVhJxB+lbAczuhFEBqnwAAwIXVNypbIrucRvlhEFCtQ3swQ0IgR/q16/qlBwNaCwgnwlcpUAZjBim7kQcnIcFc0rEk4rqJDIwUAA5YoVBPVM5BKoY+EKvGJM34wOuathQk5kV16QGisKA6pHKbIFwPZ8p411S94UJTKEs/ApQWq5gzi2mJ4AJZHQSbELBg0Y29KAMHXgQePbUwflk4YiHx9bDA5q0h82FgXBLiRJiURRg5GRbI1KMeHzUFUEIcoRb1wwxKK6sDE/FYalgXxk28shyxc4BgJoCtua9RNf0RoSEmqyA4aVEsPxaUfL+JyJjtbxd02qBotGG6ThJSQCF9kFteMcY6DwQGQginMW6JyMRKxwwJBgwHX1c9i9yPmi6oEEVg07H2q/tFUhiZYzJRp8hoYKCOXAsCJdAUTkp1k5ViucQYrquYO56xlNjspz6n4waEBSBEtP3i+Z0rFFDe7goWwCc8Z9mUYDavLEdK4so5GVCbXOAE11fIDlpJzYZ6UC0SOMNMAiPOOEwRjUCAyBu8NZgbAIWlJCwkXlyBzVAdwAZj04zZthgcSIHPB76jqzJfKZBIz0ypVu2g9Y06liHbBgY1Iyk+5fCKlawlCUrGJ0OZtcyoQEQVc7ZOTsQLRpP08TTlE0QBdecGOm3SpUIVCWLu0YWLkZJ1CyfIJmfXGDhh6pCoBG5dwwDJ+d/lE5YRjS5N2NhyQmKkEYBHYppZTeB4N/opDAmHFDkw2Pm39DlAcgsy6iLW1GpqoXf2ClrowoTskfaKEbovXIdgFogxiK+YcBdyFXCOHzTOFbtPT1cUCxRkdqAtyMtSxSFZXIcNonk+5RlKc3nUqoljehMaw3Ue6ty8Xvd0BRBFB+3EWrw/RwpauuFbyKhbALpkGDpDFhE2RV7ikqiiJJHKMfB3RNmvd6GZ+Zk6ptMQlpjCTWjrgqFVp1r9MFUr2egudIbBJwwSCsMOYqxKEPeQazgVNAzQmJOVS1KuC7Vo5jiHT0Z0LeGSNp1n/UpRyVFZOXgByKpdqXqr4ZA0bvIB2z+vaukaYKmY5xijPFIAmwHg4Mr5L/mwpUyU4BhSJmYzhgevbkeBApA2gQcAPpoojTsJvzQbEiXPvcgAfcdkvkvUwVFYhR5/2tc9+/jKXr3WHM/kHEoTrc3dVhhLlLGPBd5nBhQ79ICrXBX9DFQgnjnUA7brmzMK5r2zNcoWQHWFedK5OpB0GaI9YpBeMu+IqYP3ITefaI3BYjUY5/c7c1sQooTvAD5ZRp60i2dTDnfQxVvEJUfTC2gMcpooHwrZ1DeS+5x2tubH9Ze9SZmPUO2RXSf1OZockqEBhVmZPTC2+VPXH9Kb3kKKixkIZm0TTE/ioF3Knn/h7SRTxzNly2qTpiQRcSxIGLmBhClF43BSrwMUy/sIVrHJcoawQj2B3gLmpkXBjGLCYRBuC0IEQYKCwj2tAAy4QghkoAQ6f+HatBKzkK8FEi+V2hizyoAUXkO2IDCRYb9xTAib4wVvhmAISUwzpR5WbG7YIxBEucDYtpc0Lnzh5h3tM7pEcoxFMaHTUWygh8kANfEwz6qndKHDZpu8YcAiCzMh8O1G5C6NWjSFPhGGHFZAMWYjH2IB7+mcpK6Zz5ZgGJBZc9os954sZspBRdKkEfHreTxS3t0StVcI8BDuZp+cQAqhLUsPAggl3c1/sEwV6qnLKErzcvbaE6CCLOMMPvGS1H4Uve75TJyfT8IOF7/J45hep9+DxyVWy/jOzzlt/75YP7hRZ5Aw8b/37ryI+e2N8IYmFAw7ZeRr6E6X+8AtlZz4JRAgoP38tte4ifZcRNnI45TAJlkQ2/Yd6Q9Rwk/Fq5SAMOOB9Cbgj7kQtG9ZyAqF13hMyEjiBpLKAsQEVbwUdrOaB/geCAYgRE7EMM6Aa1ad3Jng/KLhhEkFbMXh98GMqfOZIk3FBOXCDRdJCBTNX9pdv09AIQGgpbDEDnMKAPTgNgwZ1SYg7eYYAV3AnKchw4QALvNKBQEgeaREImZGFC9E9/DeFjtM8qUKGGcENoAZ7aDhHoLFnx5YYw5AveROHiiInCPBYlMOGCfEJyqOHjjE6pdIB/rhQOYBYFUhIiOORFg2AAFPQV4tYho5oeAdwAvxVhFNxBuBziY6THV5APsJxBmUEiuCEACsQLxbHiVjCYqgoGI0QJJWYENyQCJHniGjSBHXiilIhiGQWi4NRAmvTZrfBDcKgezCYhPEzISEwK8SWGNcQgZCnh2R2AWriYNrIHL3lhd+HgKZyAHkAKelxDbLwZt5ofceSN+OYeeCSfeGgdunIfGiSFlOSE94RjToFR3czj9YXAj0EEQNYh06UYGswUFMYUJqoFyakH2yzDC4ghTGYHUzAHUJ2kcwxHSYkCkXGf4UHLcs3GAhwAXcgPQQ5HIPjZDJlRv4RjNBiOyVz6gAY4AKwwCnekSE2YiOQcAH9WIi5whsuGQJ+MBojhSMg0Vey0IIk8zH+2EnVdwFj8CtFKTcPli4hAngteSYuCYepoTuOgwAl0Aa/khQfoW5+9Rb8sgqlhzvacgErEAiygIVQ0ZAuYmJN+HZDwJOG6I86dgJXYAkwRBLgRhD66EQn9BrhcgymcAUzAHXpKAElMAWQIAyDSU+xYysEsQyiAAdDUAJZCTJSJwEz4AV+AAvcgZmL8TvhcAy2oAp+YAdncAWy6QVnAAeG8AkiZ0KqiZp1JjcrtyQW0hAPFBmtyJso8WjMxjkpZysBAQA7}\n");
    sys_vgui("if {[catch {pd}]} {\n");
    sys_vgui("    proc pd {args} {pdsend [join $args \" \"]}\n");
    sys_vgui("}\n");
    sys_vgui("proc pic_ok {id} {\n");
    sys_vgui("    set vid [string trimleft $id .]\n");
    sys_vgui("    set var_name [concat var_name_$vid]\n");
    sys_vgui("    set var_outline [concat var_outline_$vid]\n");
    sys_vgui("    set var_size [concat var_size_$vid]\n");
    sys_vgui("    set var_latch [concat var_latch_$vid]\n");
    sys_vgui("    set var_snd [concat var_snd_$vid]\n");
    sys_vgui("    set var_rcv [concat var_rcv_$vid]\n");
    sys_vgui("\n");
    sys_vgui("    global $var_name\n");
    sys_vgui("    global $var_outline\n");
    sys_vgui("    global $var_size\n");
    sys_vgui("    global $var_latch\n");
    sys_vgui("    global $var_snd\n");
    sys_vgui("    global $var_rcv\n");
    sys_vgui("\n");
    sys_vgui("    set cmd [concat $id ok \\\n");
    sys_vgui("        [string map {\" \" {\\ } \";\" \"\" \",\" \"\" \"\\\\\" \"\" \"\\{\" \"\" \"\\}\" \"\"} [eval concat $$var_name]] \\\n");
    sys_vgui("        [eval concat $$var_outline] \\\n");
    sys_vgui("        [eval concat $$var_size] \\\n");
    sys_vgui("        [eval concat $$var_latch] \\\n");
    sys_vgui("        [string map {\"$\" {\\$} \" \" {\\ } \";\" \"\" \",\" \"\" \"\\\\\" \"\" \"\\{\" \"\" \"\\}\" \"\"} [eval concat $$var_snd]] \\\n");
    sys_vgui("        [string map {\"$\" {\\$} \" \" {\\ } \";\" \"\" \",\" \"\" \"\\\\\" \"\" \"\\{\" \"\" \"\\}\" \"\"} [eval concat $$var_rcv]] \\;]\n");
    sys_vgui("    pd $cmd\n");
    sys_vgui("    pic_cancel $id\n");
    sys_vgui("}\n");
    sys_vgui("proc pic_cancel {id} {\n");
    sys_vgui("    set cmd [concat $id cancel \\;]\n");
    sys_vgui("    pd $cmd\n");
    sys_vgui("}\n");
    sys_vgui("proc pic_properties {id name outline size latch snd rcv} {\n");
    sys_vgui("    set vid [string trimleft $id .]\n");
    sys_vgui("    set var_name [concat var_name_$vid]\n");
    sys_vgui("    set var_outline [concat var_outline_$vid]\n");
    sys_vgui("    set var_size [concat var_size_$vid]\n");
    sys_vgui("    set var_latch [concat var_latch_$vid]\n");
    sys_vgui("    set var_snd [concat var_snd_$vid]\n");
    sys_vgui("    set var_rcv [concat var_rcv_$vid]\n");
    sys_vgui("\n");
    sys_vgui("    global $var_name\n");
    sys_vgui("    global $var_outline\n");
    sys_vgui("    global $var_size\n");
    sys_vgui("    global $var_latch\n");
    sys_vgui("    global $var_snd\n");
    sys_vgui("    global $var_rcv\n");
    sys_vgui("\n");
    sys_vgui("    set $var_name [string map {{\\ } \" \"} $name]\n"); // remove escape from space
    sys_vgui("    set $var_outline $outline\n");
    sys_vgui("    set $var_size $size\n");
    sys_vgui("    set $var_latch $latch\n");
    sys_vgui("    set $var_snd [string map {{\\ } \" \"} $snd]\n"); // remove escape from space
    sys_vgui("    set $var_rcv [string map {{\\ } \" \"} $rcv]\n"); // remove escape from space
    sys_vgui("\n");
    sys_vgui("    toplevel $id\n");
    sys_vgui("    wm title $id {[pic] Properties}\n");
    sys_vgui("    wm protocol $id WM_DELETE_WINDOW [concat pic_cancel $id]\n");
    sys_vgui("\n");
    sys_vgui("    frame $id.pic\n");
    sys_vgui("    pack $id.pic -side top\n");
    sys_vgui("    label $id.pic.lname -text \"File Name:\"\n");
    sys_vgui("    entry $id.pic.name -textvariable $var_name -width 30\n");
    sys_vgui("    label $id.pic.loutline -text \"Outline:\"\n");
    sys_vgui("    checkbutton $id.pic.outline -variable $var_outline \n");
    sys_vgui("    pack $id.pic.lname $id.pic.name $id.pic.loutline $id.pic.outline -side left\n");
    sys_vgui("\n");
    sys_vgui("    frame $id.sz_latch\n");
    sys_vgui("    pack $id.sz_latch -side top\n");
    sys_vgui("    label $id.sz_latch.lsize -text \"Report Size:\"\n");
    sys_vgui("    checkbutton $id.sz_latch.size -variable $var_size \n");
    sys_vgui("    label $id.sz_latch.llatch -text \"Latch Mode:\"\n");
    sys_vgui("    checkbutton $id.sz_latch.latch -variable $var_latch \n");
    sys_vgui("    pack $id.sz_latch.lsize $id.sz_latch.size $id.sz_latch.llatch $id.sz_latch.latch -side left\n");
    sys_vgui("\n");
    sys_vgui("    frame $id.snd_rcv\n");
    sys_vgui("    pack $id.snd_rcv -side top\n");
    sys_vgui("    label $id.snd_rcv.lsnd -text \"Send symbol:\"\n");
    sys_vgui("    entry $id.snd_rcv.snd -textvariable $var_snd -width 12\n");
    sys_vgui("    label $id.snd_rcv.lrcv -text \"Receive symbol:\"\n");
    sys_vgui("    entry $id.snd_rcv.rcv -textvariable $var_rcv -width 12\n");
    sys_vgui("    pack $id.snd_rcv.lsnd $id.snd_rcv.snd $id.snd_rcv.lrcv $id.snd_rcv.rcv -side left\n");
    sys_vgui("\n");
    sys_vgui("    frame $id.buttonframe\n");
    sys_vgui("    pack $id.buttonframe -side bottom -fill x -pady 2m\n");
    sys_vgui("    button $id.buttonframe.cancel -text {Cancel} -command \"pic_cancel $id\"\n");
    sys_vgui("    button $id.buttonframe.ok -text {OK} -command \"pic_ok $id\"\n");
    sys_vgui("    pack $id.buttonframe.cancel -side left -expand 1\n");
    sys_vgui("    pack $id.buttonframe.ok -side left -expand 1\n");
    sys_vgui("}\n");
}
