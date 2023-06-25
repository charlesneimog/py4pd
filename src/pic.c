#include "py4pd.h"

t_widgetbehavior py4pd_widgetbehavior;
static t_class *PY4PD_edit_proxy_class;

// =================================================

void PY4PD_draw_io_let(t_py *x){
    // check if it is on edit mode
    t_canvas *cv = glist_getcanvas(x->x_glist);
    if (cv->gl_edit == 0){
        return;
    }

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
int PY4PD_click(t_py *object, struct _glist *glist, int xpos, int ypos, int shift, int alt, int dbl, int doit){
    (void)object;
    (void)glist;
    (void)xpos;
    (void)ypos;
    

    if(dbl){

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
    sys_vgui(".x%lx.c move %lx_in %d %d\n", cv, obj, dx * obj->x_zoom, dy * obj->x_zoom);
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
    t_py *x = (t_py *)z;
    canvas_deletelinesfor(glist, (t_text *)z);
    t_canvas *cv = glist_getcanvas(x->x_glist);
    sys_vgui(".x%lx.c delete %lx_in\n", cv, x);
    sys_vgui(".x%lx.c delete %lx_out\n", cv, x);
}


// =================================================
void PY4PD_draw(t_py* x, struct _glist *glist, t_floatarg vis){

    t_canvas *cv = glist_getcanvas(glist);
    int xpos = text_xpix(&x->x_obj, x->x_glist), ypos = text_ypix(&x->x_obj, x->x_glist);
    int visible = (glist_isvisible(x->x_glist) && gobj_shouldvis((t_gobj *)x, x->x_glist));
    if(x->x_def_img && (visible || vis)){ // TODO: REMOVE THIS, THERE IS JUST ONE STATIC IMAGE
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
        if(!x->x_init){
            x->x_init = 1;
        }
        else if((visible || vis)){
            sys_vgui("if { [info exists %lx_picname] == 1 } {.x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d}\n",
                x->x_fullname, cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
        }
    }
    sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d\n", 
                        cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);

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
    if (vis){
        PY4PD_draw(x, glist, 1);
    }
    else{
        PY4PD_erase(x, glist);
    }
    return;
}

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

// =================================================
void PY4PD_edit_proxy_any(t_py4pd_edit_proxy *p, t_symbol *s, int ac, t_atom *av){
    int edit = ac = 0;
    
    if (s == gensym("key") && p->p_cnv->mouseIsOver){
        if ((av + 0)->a_w.w_float == 1){
            if ((av + 1)->a_w.w_float == 80 || (av + 1)->a_w.w_float == 112){
                post("Playing Object %p", p->p_cnv);
                // TODO: Add function to play object, Scores, for example, must send to the outputs some datas...
            }
            else if ((av + 1)->a_w.w_float == 83 || (av + 1)->a_w.w_float == 115){
                post("Show Object %p", p->p_cnv);
                // TODO: This function must show the object, open some Python Window, for example...
            }
        }
        return;
    }
    else if (s == gensym("motion")) {
        int mouse_x = (av+0)->a_w.w_float; // horizontal coordinate
        int mouse_y = (av+1)->a_w.w_float; // vertical coordinate 

        // Get object position
        int obj_x = p->p_cnv->x_obj.te_xpix * p->p_cnv->x_zoom;
        int obj_y = p->p_cnv->x_obj.te_ypix * p->p_cnv->x_zoom;

        // Get object size
        int obj_width = p->p_cnv->x_width;
        int obj_height = p->p_cnv->x_height;
        
        // Check if the mouse is over the object
        if (mouse_x >= obj_x && mouse_x <= (obj_x + obj_width) &&
            mouse_y >= obj_y && mouse_y <= (obj_y + obj_height)) {
                p->p_cnv->mouseIsOver = 1;
        }
        else{
            p->p_cnv->mouseIsOver = 0;
        }
        return;
    }

    if(p->p_cnv){

        if(s == gensym("editmode")){
            edit = (int)(av->a_w.w_float);
        }
        else if(s == gensym("obj") || s == gensym("msg") || s == gensym("floatatom")
        || s == gensym("symbolatom") || s == gensym("text") || s == gensym("bng")
        || s == gensym("toggle") || s == gensym("numbox") || s == gensym("vslider")
        || s == gensym("hslider") || s == gensym("vradio") || s == gensym("hradio")
        || s == gensym("vumeter") || s == gensym("mycnv") || s == gensym("selectall")){
            edit = 1;
        }
        else{
            return;
        }

        if(p->p_cnv->x_edit != edit){
            p->p_cnv->x_edit = edit;
            t_canvas *cv = glist_getcanvas(p->p_cnv->x_glist);
            if(edit){
                int x = text_xpix(&p->p_cnv->x_obj, p->p_cnv->x_glist);
                int y = text_ypix(&p->p_cnv->x_obj, p->p_cnv->x_glist);
                int w = p->p_cnv->x_width;
                int h = p->p_cnv->x_height;
                if(!p->p_cnv->x_outline){
                    sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d\n",
                             cv, x, y, x+w, y+h, p->p_cnv, p->p_cnv->x_zoom);
                }
                PY4PD_draw_io_let(p->p_cnv);
            }
            else{
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

    if(x->x_receive != &s_){
        pd_unbind(&x->x_obj.ob_pd, x->x_receive);
    }
    pd_unbind(&x->x_obj.ob_pd, x->x_x);
    clock_delay(x->x_proxy->p_clock, 0);
    gfxstub_deleteforkey(x);
}

// =====================================
void py4pd_picDefintion(t_py *x) {
    if (x->x_width == 250 && x->x_height == 250) {
        if (x->visMode == 1) {
            sys_vgui("image create photo PY4PD_def_img_CANVAS -data {%s} \n", PY4PD_IMAGE); // BUG: this is because I have diferentes images
        } 
        else if (x->visMode == 2) {
            sys_vgui("image create photo PY4PD_def_img_PIC -data {%s} \n", PY4PD_IMAGE); // TODO: THING if I need to use special images for something
        }
        else if (x->visMode == 3) {
            sys_vgui("image create photo PY4PD_def_img_SCORE -data {%s} \n", PY4PD_IMAGE);
        }
    }
    else{
        sys_vgui("image create photo PY4PD_def_img_WHITE -data {%s} \n", PY4PD_IMAGE);
        x->visMode = 4;
    }
  
    sys_vgui("if {[catch {pd}]} {\n");
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
                       int argc, t_atom *argv, t_class *obj_class) {
    if (py4pdArgs == gensym("-canvas")) {
        x->visMode = 1;
    }
    else if (py4pdArgs == gensym("-picture") || py4pdArgs == gensym("-pic")) {
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
        class_setwidget(obj_class, &py4pd_widgetbehavior);
    }

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
    int loaded = x->x_def_img = x->x_init = x->x_latch = 0;
    x->x_outline = x->x_size = 0;
    x->x_fullname = NULL;
    x->x_edit = c->gl_edit;

    if (!loaded) {  // default image
        x->x_def_img = 1;
    }
    if (x->x_width == 0 && x->x_height == 0) {
        x->x_width = 250;
        x->x_height = 250;
    }

    int j;
    for (j = index; j < argc; j++) {
        argv[j] = argv[j + 1];
    }
    argc--;
    py4pd_picDefintion(x);
}


