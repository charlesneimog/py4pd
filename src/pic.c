#include "m_pd.h"
#include "py4pd.h"

t_widgetbehavior py4pd_widgetbehavior;
static t_class *PY4PD_edit_proxy_class;

// =================================================

void Py4pdPic_DrawIOLet(t_py *x){
    // check if it is on edit mode
    t_canvas *cv = glist_getcanvas(x->glist);
    int xpos = text_xpix(&x->obj, x->glist), ypos = text_ypix(&x->obj, x->glist);
    
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
const char* Py4pdPic_Filepath(t_py *x, const char *filename){
    static char fn[MAXPDSTRING];
    char *bufptr;
    int fd = canvas_open(glist_getcanvas(x->glist),
        filename, "", fn, &bufptr, MAXPDSTRING, 1);
    if(fd > 0){
        fn[strlen(fn)]='/';
        sys_close(fd);
        return(fn);
    }
    else{
        sys_close(fd);
        pd_error(x, "[py4pd] can't open file %s", filename);
        return NULL;
    }
}

// =================================================
int Py4pdPic_Click(t_py *object, struct _glist *glist, int xpos, int ypos, int shift, int alt, int dbl, int doit){
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
void Py4pdPic_GetRect(t_gobj *z, t_glist *glist, int *xp1, int *yp1, int *xp2, int *yp2){
    t_py* x = (t_py*)z;
    int xpos = *xp1 = text_xpix(&x->obj, glist), ypos = *yp1 = text_ypix(&x->obj, glist);
    *xp2 = xpos + x->x_width, *yp2 = ypos + x->x_height;
}

// =================================================
void Py4pdPic_Displace(t_gobj *z, t_glist *glist, int dx, int dy){
    t_py *obj = (t_py *)z;
    obj->obj.te_xpix += dx, obj->obj.te_ypix += dy;
    t_canvas *cv = glist_getcanvas(glist);
    sys_vgui(".x%lx.c move %lx_outline %d %d\n", cv, obj, dx * obj->x_zoom, dy * obj->x_zoom);
    sys_vgui(".x%lx.c move %lx_picture %d %d\n", cv, obj, dx * obj->x_zoom, dy * obj->x_zoom);
    sys_vgui(".x%lx.c move %lx_in %d %d\n", cv, obj, dx * obj->x_zoom, dy * obj->x_zoom);
    sys_vgui(".x%lx.c move %lx_out %d %d\n", cv, obj, dx * obj->x_zoom, dy * obj->x_zoom);
    canvas_fixlinesfor(glist, (t_text*)obj);
}


// =================================================
void Py4pdPic_Select(t_gobj *z, t_glist *glist, int state){
    t_py *x = (t_py *)z;
    int xpos = text_xpix(&x->obj, glist);
    int ypos = text_ypix(&x->obj, glist);
    t_canvas *cv = glist_getcanvas(glist);
    x->x_sel = state;
    if(state){
        // sys_vgui(".x%lx.c delete %lx_outline\n", cv, x);
        sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline blue -width %d\n",
            cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
     }
    else{
        // sys_vgui(".x%lx.c delete %lx_outline\n", cv, x);
        if(x->x_edit)
            sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d\n",
                cv, xpos, ypos, xpos+x->x_width, ypos+x->x_height, x, x->x_zoom);
    }
}


// =================================================
void Py4pdPic_Delete(t_gobj *z, t_glist *glist){
    t_py *x = (t_py *)z;
    canvas_deletelinesfor(glist, (t_text *)z);
    t_canvas *cv = glist_getcanvas(x->glist);
    
    sys_vgui("if {[info exists %lx_in] == 1} {delete %lx_in}\n", cv, x);
    sys_vgui("if {[info exists %lx_out] == 1} {delete %lx_out}\n", cv, x);
}


// =================================================
void Py4pdPic_Draw(t_py* x, struct _glist *glist, t_floatarg vis){

    t_canvas *cv = glist_getcanvas(glist);
    int xpos = text_xpix(&x->obj, x->glist), ypos = text_ypix(&x->obj, x->glist);
    int visible = (glist_isvisible(x->glist) && gobj_shouldvis((t_gobj *)x, x->glist));
    if(x->x_def_img && (visible || vis)){ // TODO: REMOVE THIS, THERE IS JUST ONE STATIC IMAGE
        sys_vgui(".x%lx.c create image %d %d -anchor nw -tags %lx_picture\n", cv, xpos, ypos, x);
        sys_vgui(".x%lx.c itemconfigure %lx_picture -image PY4PD_IMAGE_{%p}\n", cv, x, x);
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

    Py4pdPic_DrawIOLet(x);
}

// =================================================
void Py4pdPic_Erase(t_py* x, struct _glist *glist){
    t_canvas *cv = glist_getcanvas(glist);
    sys_vgui(".x%lx.c delete %lx_picture\n", cv, x); // ERASE
    sys_vgui(".x%lx.c delete %lx_in\n", cv, x);
    sys_vgui(".x%lx.c delete %lx_out\n", cv, x);
    sys_vgui(".x%lx.c delete %lx_outline\n", cv, x); // if edit?
}

// =================================================
void Py4pdPic_Vis(t_gobj *z, t_glist *glist, int vis){
    t_py* x = (t_py*)z;
    if (vis){
        Py4pdPic_Draw(x, glist, 1);
    }
    else{
        Py4pdPic_Erase(x, glist);
    }
    return;
}

// ==========================================================
void Py4pdPic_Zoom(t_py *x, t_floatarg zoom){
    x->x_zoom = (int)zoom;
}

// =================================================
void Py4pdPic_EditProxyAny(t_py4pd_edit_proxy *p, t_symbol *s, int ac, t_atom *av){
    int edit = ac = 0;


    if (p->p_cnv->x_drawIOlets){
        Py4pdPic_DrawIOLet(p->p_cnv);
        p->p_cnv->x_drawIOlets = 0;
    }
    
    if (s == gensym("key") && p->p_cnv->mouseIsOver){
        if ((av + 0)->a_w.w_float == 1){
            if ((av + 1)->a_w.w_float == 80 || (av + 1)->a_w.w_float == 112){ // p or P will play the object
                post("Playing Object %p", p->p_cnv);
                // TODO: Add function to play object, Scores, for example, must send to the outputs some datas...
            }
            else if ((av + 1)->a_w.w_float == 83 || (av + 1)->a_w.w_float == 115){ // s or S will show the object
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
        int obj_x = p->p_cnv->obj.te_xpix * p->p_cnv->x_zoom;
        int obj_y = p->p_cnv->obj.te_ypix * p->p_cnv->x_zoom;

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
            t_canvas *cv = glist_getcanvas(p->p_cnv->glist);
            if(edit){
                int x = text_xpix(&p->p_cnv->obj, p->p_cnv->glist);
                int y = text_ypix(&p->p_cnv->obj, p->p_cnv->glist);
                int w = p->p_cnv->x_width;
                int h = p->p_cnv->x_height;
                sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline black -width %d\n",
                        cv, x, y, x+w, y+h, p->p_cnv, p->p_cnv->x_zoom);
            }
        }
    }
}

// ==================== PY4PD_edit_proxy ====================
void Py4pdPic_EditProxyFree(t_py4pd_edit_proxy *p){
    pd_unbind(&p->p_obj.ob_pd, p->p_sym);
    clock_free(p->p_clock);
    pd_free(&p->p_obj.ob_pd);
}

// ==========================================================
t_py4pd_edit_proxy *Py4pdPic_EditProxyNew(t_py *x, t_symbol *s){
    t_py4pd_edit_proxy *p = (t_py4pd_edit_proxy*)pd_new(PY4PD_edit_proxy_class);
    p->p_cnv = x;
    pd_bind(&p->p_obj.ob_pd, p->p_sym = s);
    p->p_clock = clock_new(p, (t_method)Py4pdPic_EditProxyFree);
    return(p);
}

// ==========================================================
void Py4pdPic_Free(t_py *x){ // delete if variable is unset and image is unused
    sys_vgui("if { [info exists %lx_picname] == 1 && [image inuse %lx_picname] == 0} { image delete %lx_picname \n unset %lx_picname\n}\n",
        x->x_fullname, x->x_fullname, x->x_fullname, x->x_fullname);

    if (strcmp(x->x_image, PY4PD_IMAGE) != 0){
        freebytes(x->x_image, strlen(x->x_image));
    }

    // check if is necessary to pd_unbind()
    if (x->x_proxy){
        pd_unbind(&x->obj.ob_pd, x->x_x);
    }
    
    clock_delay(x->x_proxy->p_clock, 0);
    gfxstub_deleteforkey(x);
}

// =====================================
void Py4pdPic_PicDefinition(t_py *x) {
    x->visMode = 1;
    sys_vgui("image create photo PY4PD_IMAGE_{%p} -data {%s} \n", x, x->x_image);
  
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
void Py4pdPic_InitVisMode(t_py *x, t_canvas *c, t_symbol *py4pdArgs, int index,
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
    class_addanything(PY4PD_edit_proxy_class, Py4pdPic_EditProxyAny);
    py4pd_widgetbehavior.w_getrectfn = Py4pdPic_GetRect;
    py4pd_widgetbehavior.w_displacefn = Py4pdPic_Displace;
    py4pd_widgetbehavior.w_selectfn = Py4pdPic_Select;
    py4pd_widgetbehavior.w_deletefn = Py4pdPic_Delete;
    py4pd_widgetbehavior.w_visfn = Py4pdPic_Vis;
    py4pd_widgetbehavior.w_clickfn = (t_clickfn)Py4pdPic_Click;

    class_setwidget(obj_class, &py4pd_widgetbehavior);
    t_canvas *cv = canvas_getcurrent();
    x->glist = (t_glist *)cv;
    x->x_zoom = x->glist->gl_zoom;
    char buf[MAXPDSTRING];
    #ifdef _WIN64
        snprintf(buf, MAXPDSTRING - 1, ".x%llx", (uintptr_t)cv);
    #else
        snprintf(buf, MAXPDSTRING - 1, ".x%lx", (unsigned long)cv);
    #endif

    buf[MAXPDSTRING - 1] = 0;
    x->x_proxy = Py4pdPic_EditProxyNew(x, gensym(buf));
    

    #ifdef _WIN64
        snprintf(buf, MAXPDSTRING - 1, ".x%llx.c", (uintptr_t)cv);
    #else
        snprintf(buf, MAXPDSTRING - 1, ".x%lx.c", (unsigned long)cv);   
    #endif
    x->x_x = gensym(buf);

    pd_bind(&x->obj.ob_pd, x->x_x);

    x->x_edit = cv->gl_edit;
    x->x_filename = &s_;
    int loaded = x->x_def_img = x->x_init = 0;
    x->x_fullname = NULL;
    x->x_edit = c->gl_edit;

    if (!loaded) {  // default image
        x->x_def_img = 1;
    }
    x->x_drawIOlets = 1;
    if (x->x_width == 0 && x->x_height == 0) {
        x->x_width = 250;
        x->x_height = 250;
    }

    int j;
    for (j = index; j < argc; j++) {
        argv[j] = argv[j + 1];
    }
    argc--;
    Py4pdPic_PicDefinition(x);
}


