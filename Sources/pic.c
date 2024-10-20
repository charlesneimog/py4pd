#include "pic.h"
#include "py4pd.h"

t_widgetbehavior Py4pdWidget;
static t_class *Py4pdEditProxy;

// =================================================
/**
 * @brief It draws the inlets and outlets of the object.
 * @param x is the py4pd object
 * @return void
 */
void Py4pdPic_DrawIOLet(t_py *x) {
    t_canvas *cv = glist_getcanvas(x->Glist);
    int xpos = text_xpix(&x->obj, x->Glist);
    int ypos = text_ypix(&x->obj, x->Glist);
    sys_vgui(".x%lx.c delete %lx_in\n", cv, x);
    sys_vgui(".x%lx.c delete %lx_out\n", cv, x);
    sys_vgui(".x%lx.c create rectangle %d %d %d %d -fill black -tags %lx_in\n", cv, xpos, ypos,
             xpos + (IOWIDTH * x->Zoom), ypos + (IHEIGHT * x->Zoom), x); // inlet 1
    if (x->nInlets == 2) {
        int inlet_width = IOWIDTH * x->Zoom;
        sys_vgui(".x%lx.c create rectangle %d %d %d %d -fill black -tags %lx_in\n", cv,
                 xpos + x->Width - inlet_width, ypos, xpos + x->Width, ypos + IHEIGHT * x->Zoom, x);
    } else if (x->nInlets > 2) {
        int inlet_width = IOWIDTH * x->Zoom;
        int ghostLines = x->Width / (x->nInlets - 1);
        int i;
        for (i = 1; i < x->nInlets - 1; i++) {
            sys_vgui(".x%lx.c create rectangle %d %d %d %d -fill black -tags "
                     "%lx_in\n",
                     cv,                                          // canvas
                     xpos + (i * ghostLines) - (inlet_width / 2), // size horizontal
                     ypos,                                        // size vertical
                     xpos + (i * ghostLines) + (inlet_width / 2), // size horizontal
                     ypos + IHEIGHT * x->Zoom,                    // size vertical
                     x);                                          // object
        }
        sys_vgui(".x%lx.c create rectangle %d %d %d %d -fill black -tags %lx_in\n", cv,
                 xpos + x->Width - inlet_width, ypos, xpos + x->Width, ypos + IHEIGHT * x->Zoom, x);
    }

    sys_vgui(".x%lx.c create rectangle %d %d %d %d -fill black -tags %lx_out\n", cv, xpos,
             ypos + x->Height, xpos + IOWIDTH * x->Zoom, ypos + x->Height - IHEIGHT * x->Zoom, x);
}

// =================================================
/**
 * @brief It returns the filepath for the figure
 * @param x is the py4pd object
 * @param filename is the name of the file
 * @return char*
 */
const char *Py4pdPic_Filepath(t_py *x, const char *filename) {
    static char fn[MAXPDSTRING];
    char *bufptr;
    int fd = canvas_open(glist_getcanvas(x->Glist), filename, "", fn, &bufptr, MAXPDSTRING, 1);
    if (fd > 0) {
        fn[strlen(fn)] = '/';
        sys_close(fd);
        return (fn);
    } else {
        sys_close(fd);
        pd_error(x, "[py4pd] can't open file %s", filename);
        return NULL;
    }
}

// =================================================
/**
 * @brief It can configure the object when it is clicked
 * @return int not used
 */
int Py4pdPic_Click(t_py *object, struct _glist *glist, int xpos, int ypos, int shift, int alt,
                   int dbl, int doit) {
    (void)object;
    (void)glist;
    (void)xpos;
    (void)ypos;

    if (dbl) {
    }

    if (alt) {
        // TODO: post("[py4pd] Alt or Option click on object");
    }

    if (shift) {
        // TODO: post("[py4pd] Shift click on object");
    }

    if (doit) {
        // TODO: post("[py4pd] Click on object");
    }

    return 1;
}

// =================================================
/**
 * @brief It returns the position of the object
 * @param z is the py4pd object
 * @param glist is the glist
 * @param xp1 is the x position
 * @param yp1 is the y position
 * @param xp2 is the x position
 * @param yp2 is the y position
 */
void Py4pdPic_GetRect(t_gobj *z, t_glist *glist, int *xp1, int *yp1, int *xp2, int *yp2) {
    t_py *x = (t_py *)z;
    int xpos = *xp1 = text_xpix(&x->obj, glist), ypos = *yp1 = text_ypix(&x->obj, glist);
    *xp2 = xpos + x->Width, *yp2 = ypos + x->Height;
}

// =================================================
/**
 * @brief It displace the object
 * @param z is the py4pd object
 * @param glist is the glist
 * @param dx is the x position
 * @param dy is the y position
 */
void Py4pdPic_Displace(t_gobj *z, t_glist *glist, int dx, int dy) {
    t_py *obj = (t_py *)z;
    obj->obj.te_xpix += dx, obj->obj.te_ypix += dy;
    t_canvas *cv = glist_getcanvas(glist);
    sys_vgui(".x%lx.c move %lx_outline %d %d\n", cv, obj, dx * obj->Zoom, dy * obj->Zoom);
    sys_vgui(".x%lx.c move %lx_picture %d %d\n", cv, obj, dx * obj->Zoom, dy * obj->Zoom);
    sys_vgui(".x%lx.c move %lx_in %d %d\n", cv, obj, dx * obj->Zoom, dy * obj->Zoom);
    sys_vgui(".x%lx.c move %lx_out %d %d\n", cv, obj, dx * obj->Zoom, dy * obj->Zoom);
    canvas_fixlinesfor(glist, (t_text *)obj);
}

// =================================================
/**
 * @brief Runned when the object is selected
 * @param z is the py4pd object
 * @param glist is the glist
 * @param state is the state
 */
void Py4pdPic_Select(t_gobj *z, t_glist *glist, int state) {
    t_py *x = (t_py *)z;
    int xpos = text_xpix(&x->obj, glist);
    int ypos = text_ypix(&x->obj, glist);
    t_canvas *cv = glist_getcanvas(glist);
    if (state) {
        // sys_vgui(".x%lx.c delete %lx_outline\n", cv, x);
        sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline "
                 "-outline blue -width %d\n",
                 cv, xpos, ypos, xpos + x->Width, ypos + x->Height, x, x->Zoom);
    } else {
        // sys_vgui(".x%lx.c delete %lx_outline\n", cv, x);
        if (x->Edit)
            sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline "
                     "-outline black -width %d\n",
                     cv, xpos, ypos, xpos + x->Width, ypos + x->Height, x, x->Zoom);
    }
}

// =================================================
/**
 * @brief It deletes the object GUI (rects)
 * @param z is the py4pd object
 * @param glist is the glist
 */
void Py4pdPic_Delete(t_gobj *z, t_glist *glist) {
    t_py *x = (t_py *)z;
    canvas_deletelinesfor(glist, (t_text *)z);
    t_canvas *cv = glist_getcanvas(x->Glist);
    sys_vgui("if {[info exists %lx_in] == 1} {delete %lx_in}\n", cv, x);
    sys_vgui("if {[info exists %lx_out] == 1} {delete %lx_out}\n", cv, x);
}

// =================================================
/**
 * @brief It draws the object, called when the object is created
 * @param z is the py4pd object
 * @param glist is the glist
 * @param vis is the visibility
 */
void Py4pdPic_Draw(t_py *x, struct _glist *glist, t_floatarg vis) {

    t_canvas *cv = glist_getcanvas(glist);
    int xpos = text_xpix(&x->obj, x->Glist), ypos = text_ypix(&x->obj, x->Glist);
    int visible = (glist_isvisible(x->Glist) && gobj_shouldvis((t_gobj *)x, x->Glist));
    if (x->DefImg && (visible || vis)) {
        sys_vgui(".x%lx.c create image %d %d -anchor nw -tags %lx_picture\n", cv, xpos, ypos, x);
        sys_vgui(".x%lx.c itemconfigure %lx_picture -image PY4PD_IMAGE_{%p}\n", cv, x, x);
    } else {
        if (visible || vis) {
            sys_vgui("if { [info exists %lx_picname] == 1 } { .x%lx.c create image "
                     "%d %d -anchor nw -image %lx_picname -tags %lx_picture\n} \n",
                     x->PicFilePath, cv, xpos, ypos, x->PicFilePath, x);
        }

        if ((visible || vis)) {
            sys_vgui("if { [info exists %lx_picname] == 1 } {.x%lx.c create "
                     "rectangle %d %d %d %d -tags %lx_outline -outline black "
                     "-width %d}\n",
                     x->PicFilePath, cv, xpos, ypos, xpos + x->Width, ypos + x->Height, x, x->Zoom);
        }
    }
    sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline -outline "
             "black -width %d\n",
             cv, xpos, ypos, xpos + x->Width, ypos + x->Height, x, x->Zoom);

    Py4pdPic_DrawIOLet(x);
}

// =================================================
/**
 * @brief It deletes the object GUI when create new or resize
 * @param z is the py4pd object
 * @param glist is the glist
 */
void Py4pdPic_ErasePic(t_py *x, struct _glist *glist) {
    t_canvas *cv = glist_getcanvas(glist);
    sys_vgui(".x%lx.c delete %lx_picture\n", cv, x); // ERASE
    sys_vgui(".x%lx.c delete %lx_in\n", cv, x);
    sys_vgui(".x%lx.c delete %lx_out\n", cv, x);
    sys_vgui(".x%lx.c delete %lx_outline\n", cv, x); // if edit?
}

// =================================================
/**
 * @brief It check if we need or not to draw the object
 */
void Py4pdPic_Vis(t_gobj *z, t_glist *glist, int vis) {
    t_py *x = (t_py *)z;
    if (vis) {
        Py4pdPic_Draw(x, glist, 1);
    } else {
        Py4pdPic_ErasePic(x, glist);
    }
    return;
}

// ==========================================================
/**
 * @brief It set the zoom when it is changed
 */
void Py4pdPic_Zoom(t_py *x, t_floatarg zoom) { x->Zoom = (int)zoom; }

// =================================================
/**
 * @brief This function is called for keys, motions, etc.
 * @param p is the py4pd object
 * @param s is the symbol
 * @param ac is the number of arguments
 * @param av is the array of arguments
 */
void Py4pdPic_EditProxyAny(t_py4pd_edit_proxy *p, t_symbol *s, int ac, t_atom *av) {

    (void)ac;
    int edit = 0;

    if (p->p_cnv->DrawIOlets) {
        Py4pdPic_DrawIOLet(p->p_cnv);
        p->p_cnv->DrawIOlets = 0;
    }

    if (s == gensym("key") && p->p_cnv->MouseIsOver) {
        if ((av + 0)->a_w.w_float == 1) {
            if ((av + 1)->a_w.w_float == 80 ||
                (av + 1)->a_w.w_float == 112) { // p or P will play the object
                post("Playing Object %p", p->p_cnv);
                // TODO: FUTURE - Add function to play object, Scores, for
                // example, must send to the outputs some datas...
            } else if ((av + 1)->a_w.w_float == 83 ||
                       (av + 1)->a_w.w_float == 115) { // s or S will show the object
                post("Show Object %p", p->p_cnv);
                // TODO: FUTURE - This function must show the object, open some
                // Python Window, for example...
            }
        }
        return;
    } else if (s == gensym("motion")) {
        int mouse_x = (av + 0)->a_w.w_float; // horizontal coordinate
        int mouse_y = (av + 1)->a_w.w_float; // vertical coordinate

        // Get object position
        int obj_x = p->p_cnv->obj.te_xpix * p->p_cnv->Zoom;
        int obj_y = p->p_cnv->obj.te_ypix * p->p_cnv->Zoom;

        // Get object size
        int obj_width = p->p_cnv->Width;
        int obj_height = p->p_cnv->Height;

        // Check if the mouse is over the object
        if (mouse_x >= obj_x && mouse_x <= (obj_x + obj_width) && mouse_y >= obj_y &&
            mouse_y <= (obj_y + obj_height)) {
            p->p_cnv->MouseIsOver = 1;
            // create a rectangle around the object blue
            // sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline
            // "
            //          "-outline blue -width %d\n",
            //          p->p_cnv->glist, obj_x, obj_y, obj_x + obj_width,
            //          obj_y + obj_height, p->p_cnv, p->p_cnv->zoom);
        } else {
            p->p_cnv->MouseIsOver = 0;
            // TODO: FUTURE - Implement subinterpreters first
            // sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags %lx_outline
            // "
            //          "-outline black -width %d\n",
            //          p->p_cnv->glist, obj_x, obj_y, obj_x + obj_width,
            //          obj_y + obj_height, p->p_cnv, p->p_cnv->zoom);
        }
        return;
    }

    if (p->p_cnv) {
        if (s == gensym("editmode")) {
            edit = (int)(av->a_w.w_float);
        } else if (s == gensym("obj") || s == gensym("msg") || s == gensym("floatatom") ||
                   s == gensym("symbolatom") || s == gensym("text") || s == gensym("bng") ||
                   s == gensym("toggle") || s == gensym("numbox") || s == gensym("vslider") ||
                   s == gensym("hslider") || s == gensym("vradio") || s == gensym("hradio") ||
                   s == gensym("vumeter") || s == gensym("mycnv") || s == gensym("selectall")) {
            edit = 1;
        } else {
            return;
        }

        if (p->p_cnv->Edit != edit) {
            p->p_cnv->Edit = edit;
            t_canvas *cv = glist_getcanvas(p->p_cnv->Glist);
            if (edit) {
                int x = text_xpix(&p->p_cnv->obj, p->p_cnv->Glist);
                int y = text_ypix(&p->p_cnv->obj, p->p_cnv->Glist);
                int w = p->p_cnv->Width;
                int h = p->p_cnv->Height;
                sys_vgui(".x%lx.c create rectangle %d %d %d %d -tags "
                         "%lx_outline -outline black -width %d\n",
                         cv, x, y, x + w, y + h, p->p_cnv, p->p_cnv->Zoom);
            }
        }
    }
}

// ==================== PY4PD_edit_proxy ====================
/**
 * @brief Function to free the edit proxy object
 * @param p The edit proxy object
 */
void Py4pdPic_EditProxyFree(t_py4pd_edit_proxy *p) {
    pd_unbind(&p->p_obj.ob_pd, p->p_sym);
    clock_free(p->p_clock);
    pd_free(&p->p_obj.ob_pd);
}

// ==========================================================
/**
 * @brief Function to create the edit proxy object
 * @param x The canvas object
 * @param s The name of the symbol
 * @return The edit proxy object
 */
t_py4pd_edit_proxy *Py4pdPic_EditProxyNew(t_py *x, t_symbol *s) {
    t_py4pd_edit_proxy *p = (t_py4pd_edit_proxy *)pd_new(Py4pdEditProxy);
    p->p_cnv = x;
    pd_bind(&p->p_obj.ob_pd, p->p_sym = s);
    p->p_clock = clock_new(p, (t_method)Py4pdPic_EditProxyFree);
    return (p);
}

// ==========================================================
/**
 * @brief Function to free the object
 * @param x The object
 */
void Py4pdPic_Free(t_py *x) { // delete if variable is unset and image is unused
    sys_vgui("if { [info exists %lx_picname] == 1 && [image inuse %lx_picname] "
             "== 0} { image delete %lx_picname \n unset %lx_picname\n}\n",
             x->PicFilePath, x->PicFilePath, x->PicFilePath, x->PicFilePath);

    if (strcmp(x->ImageBase64, PY4PD_IMAGE) != 0) {
        freebytes(x->ImageBase64, strlen(x->ImageBase64));
    }
    if (x->Proxy) {
        pd_unbind(&x->obj.ob_pd, x->CanvasName);
    }
    clock_delay(x->Proxy->p_clock, 0);
    gfxstub_deleteforkey(x);
}

// =====================================
/*
 * @brief This function creates the variables for Tk
 * @param x The object
 */
void Py4pdPic_PicDefinition(t_py *x) {
    x->VisMode = 1;
    sys_vgui("image create photo PY4PD_IMAGE_{%p} -data {%s} \n", x, x->ImageBase64);

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
    sys_vgui("        [string map {\" \" {\\ } \";\" \"\" \",\" \"\" \"\\\\\" \"\" "
             "\"\\{\" \"\" \"\\}\" \"\"} [eval concat $$var_name]] \\\n");
    sys_vgui("        [eval concat $$var_outline] \\\n");
    sys_vgui("        [eval concat $$var_size] \\\n");
    sys_vgui("        [eval concat $$var_latch] \\\n");
    sys_vgui("        [string map {\"$\" {\\$} \" \" {\\ } \";\" \"\" \",\" \"\" "
             "\"\\\\\" \"\" \"\\{\" \"\" \"\\}\" \"\"} [eval concat $$var_snd]] "
             "\\\n");
    sys_vgui("        [string map {\"$\" {\\$} \" \" {\\ } \";\" \"\" \",\" \"\" "
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
    sys_vgui("    set $var_name [string map {{\\ } \" \"} $name]\n"); // remove
                                                                      // escape from
                                                                      // space
    sys_vgui("    set $var_outline $outline\n");
    sys_vgui("    set $var_size $size\n");
    sys_vgui("    set $var_latch $latch\n");
    sys_vgui("    set $var_snd [string map {{\\ } \" \"} $snd]\n"); // remove escape
                                                                    // from space
    sys_vgui("    set $var_rcv [string map {{\\ } \" \"} $rcv]\n"); // remove escape
                                                                    // from space
    sys_vgui("\n");
    sys_vgui("    toplevel $id\n");
    sys_vgui("    wm title $id {[pic] Properties}\n");
    sys_vgui("    wm protocol $id WM_DELETE_WINDOW [concat PY4PD_cancel $id]\n");
    sys_vgui("\n");
    sys_vgui("    frame $id.pic\n");
    sys_vgui("    pack $id.pic -side top\n");
    sys_vgui("    label $id.pic.lname -text \"File Name:\"\n");
    sys_vgui("    entry $id.pic.name -textvariable $var_name -width 30\n");
    sys_vgui("    label $id.pic.loutline -text \"Outline:\"\n");
    sys_vgui("    checkbutton $id.pic.outline -variable $var_outline \n");
    sys_vgui("    pack $id.pic.lname $id.pic.name $id.pic.loutline $id.pic.outline "
             "-side left\n");
    sys_vgui("\n");
    sys_vgui("    frame $id.sz_latch\n");
    sys_vgui("    pack $id.sz_latch -side top\n");
    sys_vgui("    label $id.sz_latch.lsize -text \"Report Size:\"\n");
    sys_vgui("    checkbutton $id.sz_latch.size -variable $var_size \n");
    sys_vgui("    label $id.sz_latch.llatch -text \"Latch Mode:\"\n");
    sys_vgui("    checkbutton $id.sz_latch.latch -variable $var_latch \n");
    sys_vgui("    pack $id.sz_latch.lsize $id.sz_latch.size $id.sz_latch.llatch "
             "$id.sz_latch.latch -side left\n");
    // lx_out
    sys_vgui("\n");
    sys_vgui("    frame $id.snd_rcv\n");
    sys_vgui("    pack $id.snd_rcv -side top\n");
    sys_vgui("    label $id.snd_rcv.lsnd -text \"Send symbol:\"\n");
    sys_vgui("    entry $id.snd_rcv.snd -textvariable $var_snd -width 12\n");
    sys_vgui("    label $id.snd_rcv.lrcv -text \"Receive symbol:\"\n");
    sys_vgui("    entry $id.snd_rcv.rcv -textvariable $var_rcv -width 12\n");
    sys_vgui("    pack $id.snd_rcv.lsnd $id.snd_rcv.snd $id.snd_rcv.lrcv "
             "$id.snd_rcv.rcv -side left\n");
    sys_vgui("\n");
    sys_vgui("    frame $id.buttonframe\n");
    sys_vgui("    pack $id.buttonframe -side bottom -fill x -pady 2m\n");
    sys_vgui("    button $id.buttonframe.cancel -text {Cancel} -command "
             "\"PY4PD_cancel $id\"\n");
    sys_vgui("    button $id.buttonframe.ok -text {OK} -command \"PY4PD_ok $id\"\n");
    sys_vgui("    pack $id.buttonframe.cancel -side left -expand 1\n");
    sys_vgui("    pack $id.buttonframe.ok -side left -expand 1\n");
    sys_vgui("}\n");
}

// ================================================
/**
 * @brief set vis mode and call all functions to init the vis mode
 */
void Py4pdPic_InitVisMode(t_py *x, t_canvas *c, t_symbol *py4pdArgs, int index, int argc,
                          t_atom *argv, t_class *obj_class) {

    (void)py4pdArgs; // TODO: remove this
    Py4pdEditProxy = class_new(0, 0, 0, sizeof(t_py4pd_edit_proxy), CLASS_NOINLET | CLASS_PD, 0);
    class_addanything(Py4pdEditProxy, Py4pdPic_EditProxyAny);
    Py4pdWidget.w_getrectfn = Py4pdPic_GetRect;
    Py4pdWidget.w_displacefn = Py4pdPic_Displace;
    Py4pdWidget.w_selectfn = Py4pdPic_Select;
    Py4pdWidget.w_deletefn = Py4pdPic_Delete;
    Py4pdWidget.w_visfn = Py4pdPic_Vis;
    Py4pdWidget.w_clickfn = (t_clickfn)Py4pdPic_Click;

    class_setwidget(obj_class, &Py4pdWidget);
    t_canvas *cv = canvas_getcurrent();
    x->Glist = (t_glist *)cv;
    x->Zoom = x->Glist->gl_zoom;
    char buf[MAXPDSTRING];
#ifdef _WIN64
    snprintf(buf, MAXPDSTRING - 1, ".x%llx", (uintptr_t)cv);
#else
    snprintf(buf, MAXPDSTRING - 1, ".x%lx", (unsigned long)cv);
#endif

    buf[MAXPDSTRING - 1] = 0;
    x->Proxy = Py4pdPic_EditProxyNew(x, gensym(buf));

#ifdef _WIN64
    snprintf(buf, MAXPDSTRING - 1, ".x%llx.c", (uintptr_t)cv);
#else
    snprintf(buf, MAXPDSTRING - 1, ".x%lx.c", (unsigned long)cv);
#endif

    x->CanvasName = gensym(buf);
    pd_bind(&x->obj.ob_pd, gensym(buf));
    x->Edit = cv->gl_edit;
    int loaded = x->DefImg = 0;
    x->PicFilePath = NULL;
    x->Edit = c->gl_edit;

    if (!loaded) { // default image
        x->DefImg = 1;
    }
    x->DrawIOlets = 1;
    if (x->Width == 0 && x->Height == 0) {
        x->Width = 250;
        x->Height = 250;
    }

    int j;
    for (j = index; j < argc; j++) {
        argv[j] = argv[j + 1];
    }
    argc--;
    Py4pdPic_PicDefinition(x);
}
