#include "py4pd_pic.h"

t_widgetbehavior py4pd_widgetbehavior;
static t_class *PY4PD_edit_proxy_class;

// =================================================

void PY4PD_draw_io_let(t_py *x){
    t_canvas *cv = glist_getcanvas(x->x_glist);
    int xpos = text_xpix(&x->x_obj, x->x_glist), ypos = text_ypix(&x->x_obj, x->x_glist);
    
    sys_vgui(".x%lx.c delete %lx_in\n", cv, x);
    sys_vgui(".x%lx.c delete %lx_out\n", cv, x);
    
    // =============
    // CREATE INLETS
    // =============
    sys_vgui(".x%lx.c create rectangle %d %d %d %d -fill black -tags %lx_in\n", 
            cv, xpos, ypos, xpos+(IOWIDTH * x->x_zoom), ypos+(IHEIGHT * x->x_zoom), x); // inlet 1
    if (x->x_numInlets == 2){
        int inlet_width = IOWIDTH * x->x_zoom;
        sys_vgui(".x%lx.c create rectangle %d %d %d %d -fill black -tags %lx_in\n", 
            cv, xpos + x->x_width - inlet_width, ypos, xpos + x->x_width, ypos + IHEIGHT * x->x_zoom, x);
    }
    else if (x->x_numInlets > 2){
        int inlet_width = IOWIDTH * x->x_zoom;
        int ghostLines = x->x_width /  (x->x_numInlets - 1);
        int i;
        for (i = 1; i < x->x_numInlets - 1; i++){
            sys_vgui(".x%lx.c create rectangle %d %d %d %d -fill black -tags %lx_in\n", 
                cv, // canvas
                xpos + (i * ghostLines) - (inlet_width / 2), // size horizontal
                ypos, // size vertical
                xpos + (i * ghostLines) + (inlet_width / 2), // size horizontal 
                ypos + IHEIGHT * x->x_zoom, // size vertical
                x); // object
        }
        sys_vgui(".x%lx.c create rectangle %d %d %d %d -fill black -tags %lx_in\n", 
                    cv, xpos + x->x_width - inlet_width, ypos, xpos + x->x_width, ypos + IHEIGHT * x->x_zoom, x);
    }

    // ==============
    // CREATE OUTLETS
    // ==============
    sys_vgui(".x%lx.c create rectangle %d %d %d %d -fill black -tags %lx_out\n", 
            cv, xpos, ypos+x->x_height, xpos + IOWIDTH * x->x_zoom, ypos + x->x_height - IHEIGHT * x->x_zoom, x);
}

// =================================================
const char* PY4PD_filepath(t_py *x, const char *filename){
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


// =================================================
void PY4PD_mouserelease(t_py* x){
    if(x->x_latch){
        outlet_float(x->out_A, 0);
        if(x->x_send != &s_ && x->x_send->s_thing)
            pd_float(x->x_send->s_thing, 0);
    }
}

// =================================================
// void PY4PD_get_snd_rcv(t_py* x){
    // post("I am herer");
    // t_binbuf *bb = x->x_obj.te_binbuf;
    // int n_args = binbuf_getnatom(bb), i = 0; // number of arguments
    // char buf[128];
    // if(!x->x_snd_set){ // no send set, search arguments/flags
    //     if(n_args > 0){ // we have arguments, let's search them
    //         if(x->x_flag){ // arguments are flags actually
    //             if(x->x_s_flag){ // we got a search flag, let's get it
    //                 for(i = 0;  i < n_args; i++){
    //                     atom_string(binbuf_getvec(bb) + i, buf, 80);
    //                     if(gensym(buf) == gensym("-send")){
    //                         i++;
    //                         atom_string(binbuf_getvec(bb) + i, buf, 80);
    //                         x->x_snd_raw = gensym(buf);
    //                         break;
    //                     }
    //                 }
    //             }
    //         }
    //         else{ // we got no flags, let's search for argument
    //             int arg_n = 3; // receive argument number
    //             if(n_args >= arg_n){ // we have it, get it
    //                 atom_string(binbuf_getvec(bb) + arg_n, buf, 80);
    //                 x->x_snd_raw = gensym(buf);
    //             }
    //         }
    //     }
    // }
    // if(x->x_snd_raw == &s_)
    //     x->x_snd_raw = gensym("empty");
    // if(!x->x_rcv_set){ // no receive set, search arguments
    //     if(n_args > 0){ // we have arguments, let's search them
    //         if(x->x_flag){ // arguments are flags actually
    //             if(x->x_r_flag){ // we got a receive flag, let's get it
    //                 for(i = 0;  i < n_args; i++){
    //                     atom_string(binbuf_getvec(bb) + i, buf, 80);
    //                     if(gensym(buf) == gensym("-receive")){
    //                         i++;
    //                         atom_string(binbuf_getvec(bb) + i, buf, 80);
    //                         x->x_rcv_raw = gensym(buf);
    //                         break;
    //                     }
    //                 }
    //             }
    //         }
    //         else{ // we got no flags, let's search for argument
    //             int arg_n = 4; // receive argument number
    //             if(n_args >= arg_n){ // we have it, get it
    //                 atom_string(binbuf_getvec(bb) + arg_n, buf, 80);
    //                 x->x_rcv_raw = gensym(buf);
    //             }
    //         }
    //     }
    // }
    // if(x->x_rcv_raw == &s_)
    //     x->x_rcv_raw = gensym("empty");
// }


// =================================================
int PY4PD_click(t_py *object, struct _glist *glist, int xpos, int ypos, int shift, int alt, int dbl, int doit){
    (void)object;
    (void)glist;
    (void)xpos;
    (void)ypos;

    if(dbl){
        // t_canvas *cv = glist_getcanvas(object->x_glist);
        // // int inlet_width = IOWIDTH * object->x_zoom;
        // int x_xpos = text_xpix(&object->x_obj, object->x_glist), x_ypos = text_ypix(&object->x_obj, object->x_glist);
        // 
        // // create a small green triangle in the top left corner
        //         // create a small green triangle in the top left corner rotated 90 degrees clockwise
        // int triangle_size = 10; // size of the triangle
        // int x = x_xpos; // x-coordinate of the triangle
        // int y = x_ypos; // y-coordinate of the triangle
        // // sys_vgui(".x%lx.c create polygon %d %d %d %d %d %d -fill green -tags %lx_play\n", 
        //         // cv, x + 10, y + 10, x + triangle_size + 10, y, x + 10, y+triangle_size, x);
        //


    }

    if (alt){
        // post("[py4pd] Alt or Option click on object");
    }

    if (shift){
        // post("[py4pd] Shift click on object");
    }

    if (doit){
        // post("[py4pd] Click on object");
    }

    return(1);
}

// =================================================
void PY4PD_getrect(t_gobj *z, t_glist *glist, int *xp1, int *yp1, int *xp2, int *yp2){
    t_py* x = (t_py*)z;
    int xpos = *xp1 = text_xpix(&x->x_obj, glist), ypos = *yp1 = text_ypix(&x->x_obj, glist);
    *xp2 = xpos + x->x_width, *yp2 = ypos + x->x_height;
}

// =================================================
void PY4PD_displace(t_gobj *z, t_glist *glist, int dx, int dy){
    t_py *obj = (t_py *)z;
    obj->x_obj.te_xpix += dx, obj->x_obj.te_ypix += dy;
    t_canvas *cv = glist_getcanvas(glist);
    sys_vgui(".x%lx.c move %lx_outline %d %d\n", cv, obj, dx * obj->x_zoom, dy * obj->x_zoom);
    sys_vgui(".x%lx.c move %lx_picture %d %d\n", cv, obj, dx * obj->x_zoom, dy * obj->x_zoom);
    // if(obj->x_receive == &s_) // I don't know why this is here 
    sys_vgui(".x%lx.c move %lx_in %d %d\n", cv, obj, dx * obj->x_zoom, dy * obj->x_zoom);
    // if(obj->x_send == &s_)
    sys_vgui(".x%lx.c move %lx_out %d %d\n", cv, obj, dx * obj->x_zoom, dy * obj->x_zoom);
    canvas_fixlinesfor(glist, (t_text*)obj);
}


// =================================================
void PY4PD_select(t_gobj *z, t_glist *glist, int state){
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


// =================================================
void PY4PD_delete(t_gobj *z, t_glist *glist){
    canvas_deletelinesfor(glist, (t_text *)z);
}


// =================================================
void PY4PD_draw(t_py* x, struct _glist *glist, t_floatarg vis){
    t_canvas *cv = glist_getcanvas(glist);
    int xpos = text_xpix(&x->x_obj, x->x_glist), ypos = text_ypix(&x->x_obj, x->x_glist);
    int visible = (glist_isvisible(x->x_glist) && gobj_shouldvis((t_gobj *)x, x->x_glist));
    if(x->x_def_img && (visible || vis)){ // DEFAULT PIC
        sys_vgui(".x%lx.c create image %d %d -anchor nw -tags %lx_picture\n", cv, xpos, ypos, x);
        if (x->visMode == 1) {
            sys_vgui(".x%lx.c itemconfigure %lx_picture -image %s\n", cv, x, "PY4PD_def_img_CANVAS");
        } 
        else if (x->visMode == 2) {
            sys_vgui(".x%lx.c itemconfigure %lx_picture -image %s\n", cv, x, "PY4PD_def_img_PIC");
        }
        else if (x->visMode == 3) {
            sys_vgui(".x%lx.c itemconfigure %lx_picture -image %s\n", cv, x, "PY4PD_def_img_SCORE");
        }
        else if (x->visMode == 3){
            sys_vgui(".x%lx.c itemconfigure %lx_picture -image %s\n", cv, x, "PY4PD_def_img_WHITE");
        }
    }
    else{
        if(visible || vis){
            sys_vgui("if { [info exists %lx_picname] == 1 } { .x%lx.c create image %d %d -anchor nw -image %lx_picname -tags %lx_picture\n} \n",
                x->x_fullname, cv, xpos, ypos, x->x_fullname, x);
        }
        if(!x->x_init)
            x->x_init = 1;
        else if((visible || vis)){
            sys_vgui("if { [info exists %lx_picname] == 1 } {.x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d}\n",
                x->x_fullname, cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
        }
    }
    // sys_vgui(".x%lx.c bind %lx_picture <ButtonRelease> {pdsend [concat %s _mouserelease \\;]}\n", cv, x, x->x_x->s_name);
    sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d\n", cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
    PY4PD_draw_io_let(x);
         
}

// =================================================
void PY4PD_erase(t_py* x, struct _glist *glist){
    t_canvas *cv = glist_getcanvas(glist);
    sys_vgui(".x%lx.c delete %lx_picture\n", cv, x); // ERASE
    sys_vgui(".x%lx.c delete %lx_in\n", cv, x);
    sys_vgui(".x%lx.c delete %lx_out\n", cv, x);
    sys_vgui(".x%lx.c delete %lx_outline\n", cv, x); // if edit?
}

// =================================================
void PY4PD_vis(t_gobj *z, t_glist *glist, int vis){
    t_py* x = (t_py*)z;
    vis ? PY4PD_draw(x, glist, 1) : PY4PD_erase(x, glist);
}

// =================================================
void PY4PD_save(t_gobj *z, t_binbuf *b){
    t_py *x = (t_py *)z;
    if(x->x_filename == &s_)
        x->x_filename = gensym("empty");
    // PY4PD_get_snd_rcv(x);
    binbuf_addv(b, "ssiisisssii", gensym("#X"), gensym("obj"), x->x_obj.te_xpix, x->x_obj.te_ypix,
        atom_getsymbol(binbuf_getvec(x->x_obj.te_binbuf)), x->x_outline, x->x_filename, x->x_snd_raw,
        x->x_rcv_raw, x->x_size, x->x_latch);
    binbuf_addv(b, ";");
}

// // =================================================
// void PY4PD_size_callback(t_py *x, t_float w, t_float h){ // callback
//     post("PY4PD_size_callback");
//     x->x_width = w;
//     x->x_height = h;
//     if(glist_isvisible(x->x_glist) && gobj_shouldvis((t_gobj *)x, x->x_glist)){
//         t_canvas *cv = glist_getcanvas(x->x_glist);
//         int xpos = text_xpix(&x->x_obj, x->x_glist), ypos = text_ypix(&x->x_obj, x->x_glist);
//         sys_vgui("if { [info exists %lx_picname] == 1 } { .x%lx.c create image %d %d -anchor nw -image %lx_picname -tags %lx_picture\n} \n",
//             x->x_fullname, cv, xpos, ypos, x->x_fullname, x);
//         canvas_fixlinesfor(x->x_glist, (t_text*)x);
//         if(x->x_edit || x->x_outline){
//             sys_vgui(".x%lx.c delete %lx_outline\n", cv, x);
//             if(x->x_sel)
//                 sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline blue -width %d\n",
//                 cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
//             else
//                 sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d\n",
//                 cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
//         }
//         PY4PD_draw_io_let(x);
//     }
//     else
//         PY4PD_erase(x, x->x_glist);
//     if(x->x_size){
//         t_atom at[2];
//         SETFLOAT(at, w);
//         SETFLOAT(at+1, h);
//         outlet_list(x->x_obj.ob_outlet, &s_list, 2, at);
//     }
// }
//
// =================================================
void PY4PD_open(t_py* x, t_symbol *filename){
    if(filename){
        if(filename == gensym("empty") && x->x_def_img)
            return;
        if(filename != x->x_filename){
            const char *file_name_open = PY4PD_filepath(x, filename->s_name); // path
            if(file_name_open){
                x->x_filename = filename;
                x->x_fullname = gensym(file_name_open);
                if(x->x_def_img)
                    x->x_def_img = 0;
                if(glist_isvisible(x->x_glist) && gobj_shouldvis((t_gobj *)x, x->x_glist)){
                    PY4PD_erase(x, x->x_glist);
                    sys_vgui("if {[info exists %lx_picname] == 0} {image create photo %lx_picname -file \"%s\"\n set %lx_picname 1\n}\n",
                        x->x_fullname, x->x_fullname, file_name_open, x->x_fullname);
                    PY4PD_draw(x, x->x_glist, 0);
                }
            }
            else
                pd_error(x, "[pic]: error opening file '%s'", filename->s_name);
        }
    }
    else
        pd_error(x, "[pic]: open needs a file name");
}

// // =================================================
// void PY4PD_send(t_py *x, t_symbol *s){
//     if(s != gensym("")){
//         t_symbol *snd = (s == gensym("empty")) ? &s_ : canvas_realizedollar(x->x_glist, s);
//         if(snd != x->x_send){
//             x->x_snd_raw = s;
//             x->x_send = snd;
//             x->x_snd_set = 1;
//             if(x->x_edit && glist_isvisible(x->x_glist) && gobj_shouldvis((t_gobj *)x, x->x_glist)){
//                 if(x->x_send == &s_)
//                     PY4PD_draw_io_let(x);
//                 else
//                     sys_vgui(".x%lx.c delete %lx_out\n", glist_getcanvas(x->x_glist), x);
//             }
//         }
//     }
// }
//
// void PY4PD_receive(t_py *x, t_symbol *s){
//     if(s != gensym("")){
//         t_symbol *rcv = s == gensym("empty") ? &s_ : canvas_realizedollar(x->x_glist, s);
//         if(rcv != x->x_receive){
//             if(x->x_receive != &s_)
//                 pd_unbind(&x->x_obj.ob_pd, x->x_receive);
//             x->x_rcv_set = 1;
//             x->x_rcv_raw = s;
//             x->x_receive = rcv;
//             if(x->x_receive == &s_){
//                 if(x->x_edit && glist_isvisible(x->x_glist) && gobj_shouldvis((t_gobj *)x, x->x_glist))
//                     PY4PD_draw_io_let(x);
//             }
//             else{
//                 pd_bind(&x->x_obj.ob_pd, x->x_receive);
//                 if(x->x_edit && glist_isvisible(x->x_glist) && gobj_shouldvis((t_gobj *)x, x->x_glist))
//                     sys_vgui(".x%lx.c delete %lx_in\n", glist_getcanvas(x->x_glist), x);
//             }
//         }
//     }
// }

// void PY4PD_outline(t_py *x, t_float f){
//     (void)f;
//     if(glist_isvisible(x->x_glist) && gobj_shouldvis((t_gobj *)x, x->x_glist)){
//         t_canvas *cv = glist_getcanvas(x->x_glist);
//         int xpos = text_xpix(&x->x_obj, x->x_glist), ypos = text_ypix(&x->x_obj, x->x_glist);
//         if(x->x_sel){ sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline blue -width %d\n",
//             cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
//         }
//         else{ 
//             sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d\n",
//             cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
//         }
//     }
// }


// // =================================================
// void PY4PD_size(t_py *x, t_float f){
//     int size = (int)(f != 0);
//     if(x->x_size != size)
//         x->x_size = size;
// }
//
// // =================================================
// void PY4PD_latch(t_py *x, t_float f){
//     int latch = (int)(f != 0);
//     if(x->x_latch != latch){
//         x->x_latch = latch;
//     }
// }
//
// =================================================
void PY4PD_edit_proxy_any(t_py4pd_edit_proxy *p, t_symbol *s, int ac, t_atom *av){
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
                int w = p->p_cnv->x_width;
                int h = p->p_cnv->x_height;
                if(!p->p_cnv->x_outline)
                    sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d\n",
                             cv, x, y, x+w, y+h, p->p_cnv, p->p_cnv->x_zoom);
                PY4PD_draw_io_let(p->p_cnv);
            }
            else{
                // if(!p->p_cnv->x_outline)
                    // sys_vgui(".x%lx.c delete %lx_outline\n", cv, p->p_cnv);
                sys_vgui(".x%lx.c delete %lx_in\n", cv, p->p_cnv);
                sys_vgui(".x%lx.c delete %lx_out\n", cv, p->p_cnv);
            }
        }
    }
}

// ==========================================================
void PY4PD_zoom(t_py *x, t_floatarg zoom){
    x->x_zoom = (int)zoom;
}

// ==================== PY4PD_edit_proxy ====================
void PY4PD_edit_proxy_free(t_py4pd_edit_proxy *p){
    pd_unbind(&p->p_obj.ob_pd, p->p_sym);
    clock_free(p->p_clock);
    pd_free(&p->p_obj.ob_pd);
}

// ==========================================================
t_py4pd_edit_proxy *PY4PD_edit_proxy_new(t_py *x, t_symbol *s){
    t_py4pd_edit_proxy *p = (t_py4pd_edit_proxy*)pd_new(PY4PD_edit_proxy_class);
    p->p_cnv = x;
    pd_bind(&p->p_obj.ob_pd, p->p_sym = s);
    p->p_clock = clock_new(p, (t_method)PY4PD_edit_proxy_free);
    return(p);
}

// ==========================================================
void PY4PD_free(t_py *x){ // delete if variable is unset and image is unused
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
void py4pd_picDefintion(t_py *x) {
    (void)x;

    if (x->x_width == 250 && x->x_height == 250) {
        if (x->visMode == 1) {
            sys_vgui("image create photo PY4PD_def_img_CANVAS -data {%s} \n", PY4PD_IMAGE); // BUG: this is because I have diferentes images
        } 
        else if (x->visMode == 2) {
            sys_vgui("image create photo PY4PD_def_img_PIC -data {%s} \n", PY4PD_IMAGE);
        }
        else if (x->visMode == 3) {
            sys_vgui("image create photo PY4PD_def_img_SCORE -data {%s} \n", PY4PD_SCORE);
        }
    }
    else{
        sys_vgui("image create photo PY4PD_def_img_WHITE -data {R0lGODlhKgAhAPAAAP///wAAACH5BAAAAAAAIf8LSW1hZ2VNYWdpY2sOZ2FtbWE9MC40NTQ1NDUALAAAAAAqACEAAAIkhI+py+0Po5y02ouz3rz7D4biSJbmiabqyrbuC8fyTNf2jTMFADs=} \n");
        x->visMode = 4;

    }

  
    sys_vgui("if {[catch {pd}]} {\n");
    // sys_vgui("    proc pd {args} {pdsend [join $args \" \"]}\n");
    sys_vgui("}\n");
    sys_vgui("proc PY4PD_ok {id} {\n");
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
    sys_vgui(
        "        [string map {\" \" {\\ } \";\" \"\" \",\" \"\" \"\\\\\" \"\" "
        "\"\\{\" \"\" \"\\}\" \"\"} [eval concat $$var_name]] \\\n");
    sys_vgui("        [eval concat $$var_outline] \\\n");
    sys_vgui("        [eval concat $$var_size] \\\n");
    sys_vgui("        [eval concat $$var_latch] \\\n");
    sys_vgui(
        "        [string map {\"$\" {\\$} \" \" {\\ } \";\" \"\" \",\" \"\" "
        "\"\\\\\" \"\" \"\\{\" \"\" \"\\}\" \"\"} [eval concat $$var_snd]] "
        "\\\n");
    sys_vgui(
        "        [string map {\"$\" {\\$} \" \" {\\ } \";\" \"\" \",\" \"\" "
        "\"\\\\\" \"\" \"\\{\" \"\" \"\\}\" \"\"} [eval concat $$var_rcv]] "
        "\\;]\n");
    sys_vgui("    pd $cmd\n");
    sys_vgui("    PY4PD_cancel $id\n");
    sys_vgui("}\n");
    sys_vgui("proc PY4PD_cancel {id} {\n");
    sys_vgui("    set cmd [concat $id cancel \\;]\n");
    sys_vgui("    pd $cmd\n");
    sys_vgui("}\n");
    sys_vgui("proc PY4PD_properties {id name outline size latch snd rcv} {\n");
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
    sys_vgui(
        "    set $var_name [string map {{\\ } \" \"} $name]\n");  // remove
                                                                  // escape from
                                                                  // space
    sys_vgui("    set $var_outline $outline\n");
    sys_vgui("    set $var_size $size\n");
    sys_vgui("    set $var_latch $latch\n");
    sys_vgui(
        "    set $var_snd [string map {{\\ } \" \"} $snd]\n");  // remove escape
                                                                // from space
    sys_vgui(
        "    set $var_rcv [string map {{\\ } \" \"} $rcv]\n");  // remove escape
                                                                // from space
    sys_vgui("\n");
    sys_vgui("    toplevel $id\n");
    sys_vgui("    wm title $id {[pic] Properties}\n");
    sys_vgui(
        "    wm protocol $id WM_DELETE_WINDOW [concat PY4PD_cancel $id]\n");
    sys_vgui("\n");
    sys_vgui("    frame $id.pic\n");
    sys_vgui("    pack $id.pic -side top\n");
    sys_vgui("    label $id.pic.lname -text \"File Name:\"\n");
    sys_vgui("    entry $id.pic.name -textvariable $var_name -width 30\n");
    sys_vgui("    label $id.pic.loutline -text \"Outline:\"\n");
    sys_vgui("    checkbutton $id.pic.outline -variable $var_outline \n");
    sys_vgui(
        "    pack $id.pic.lname $id.pic.name $id.pic.loutline $id.pic.outline "
        "-side left\n");
    sys_vgui("\n");
    sys_vgui("    frame $id.sz_latch\n");
    sys_vgui("    pack $id.sz_latch -side top\n");
    sys_vgui("    label $id.sz_latch.lsize -text \"Report Size:\"\n");
    sys_vgui("    checkbutton $id.sz_latch.size -variable $var_size \n");
    sys_vgui("    label $id.sz_latch.llatch -text \"Latch Mode:\"\n");
    sys_vgui("    checkbutton $id.sz_latch.latch -variable $var_latch \n");
    sys_vgui(
        "    pack $id.sz_latch.lsize $id.sz_latch.size $id.sz_latch.llatch "
        "$id.sz_latch.latch -side left\n");
    // lx_out
    sys_vgui("\n");
    sys_vgui("    frame $id.snd_rcv\n");
    sys_vgui("    pack $id.snd_rcv -side top\n");
    sys_vgui("    label $id.snd_rcv.lsnd -text \"Send symbol:\"\n");
    sys_vgui("    entry $id.snd_rcv.snd -textvariable $var_snd -width 12\n");
    sys_vgui("    label $id.snd_rcv.lrcv -text \"Receive symbol:\"\n");
    sys_vgui("    entry $id.snd_rcv.rcv -textvariable $var_rcv -width 12\n");
    sys_vgui(
        "    pack $id.snd_rcv.lsnd $id.snd_rcv.snd $id.snd_rcv.lrcv "
        "$id.snd_rcv.rcv -side left\n");
    sys_vgui("\n");
    sys_vgui("    frame $id.buttonframe\n");
    sys_vgui("    pack $id.buttonframe -side bottom -fill x -pady 2m\n");
    sys_vgui(
        "    button $id.buttonframe.cancel -text {Cancel} -command "
        "\"PY4PD_cancel $id\"\n");
    sys_vgui(
        "    button $id.buttonframe.ok -text {OK} -command \"PY4PD_ok $id\"\n");
    sys_vgui("    pack $id.buttonframe.cancel -side left -expand 1\n");
    sys_vgui("    pack $id.buttonframe.ok -side left -expand 1\n");
    sys_vgui("}\n");
}

// ================================================
void py4pd_InitVisMode(t_py *x, t_canvas *c, t_symbol *py4pdArgs, int index,
                       int argc, t_atom *argv) {
    if (py4pdArgs == gensym("-canvas")) {
        x->visMode = 1;
    }
    else if (py4pdArgs == gensym("-picture")) {
        x->visMode = 2;
    }
    else if (py4pdArgs == gensym("-score")) {
        x->visMode = 3;
    } 
    else {
        x->visMode = 1;
        pd_error(x, "[py4pd]: unknown visMode");
    }
    PY4PD_edit_proxy_class = class_new(0, 0, 0, sizeof(t_py4pd_edit_proxy), CLASS_NOINLET | CLASS_PD, 0);
    class_addanything(PY4PD_edit_proxy_class, PY4PD_edit_proxy_any);
    py4pd_widgetbehavior.w_getrectfn = PY4PD_getrect;
    py4pd_widgetbehavior.w_displacefn = PY4PD_displace;
    py4pd_widgetbehavior.w_selectfn = PY4PD_select;
    py4pd_widgetbehavior.w_deletefn = PY4PD_delete;
    py4pd_widgetbehavior.w_visfn = PY4PD_vis;
    py4pd_widgetbehavior.w_clickfn = (t_clickfn)PY4PD_click;
    if (x->pyObject == 0) {
        class_setwidget(py4pd_class_VIS, &py4pd_widgetbehavior);
    }
    else{
        class_setwidget(pyNewObject_VIS, &py4pd_widgetbehavior);
    }
    py4pd_picDefintion(x);
    t_canvas *cv = canvas_getcurrent();
    x->x_glist = (t_glist *)cv;
    x->x_zoom = x->x_glist->gl_zoom;
    char buf[MAXPDSTRING];
    #ifdef _WIN64
        snprintf(buf, MAXPDSTRING - 1, ".x%llx", (uintptr_t)cv);
    #else
        snprintf(buf, MAXPDSTRING - 1, ".x%lx", (unsigned long)cv);
    #endif
    buf[MAXPDSTRING - 1] = 0;
    x->x_proxy = PY4PD_edit_proxy_new(x, gensym(buf));
    
    #ifdef _WIN64
        snprintf(buf, MAXPDSTRING - 1, ".x%llx.c", (uintptr_t)cv);
    #else
        snprintf(buf, MAXPDSTRING - 1, ".x%lx.c", (unsigned long)cv);   
    #endif
    pd_bind(&x->x_obj.ob_pd, x->x_x = gensym(buf));
    x->x_edit = cv->gl_edit;
    x->x_send = x->x_snd_raw = x->x_receive = x->x_rcv_raw = x->x_filename = &s_;
    int loaded = x->x_rcv_set = x->x_snd_set = x->x_def_img = x->x_init = x->x_latch = 0;
    x->x_outline = x->x_size = 0;
    x->x_fullname = NULL;
    x->x_edit = c->gl_edit;
    if (!loaded) {  // default image
        x->x_width = 250;
        x->x_height = 250;
        x->x_def_img = 1;
    }
    x->x_flag = 0;
    x->x_s_flag = 0;
    x->x_r_flag = 0;
    if (x->x_receive != &s_) {
        pd_bind(&x->x_obj.ob_pd, x->x_receive);
    }

    int j;
    for (j = index; j < argc; j++) {
        argv[j] = argv[j + 1];
    }
    argc--;
    // PY4PD_draw_io_let(x);
}

