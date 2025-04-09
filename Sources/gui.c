#include "py4pd.h"

// ─────────────────────────────────────
static void pdpy_gfx_mouse_event(t_pdpy_pdobj *o, int x, int y, int type) {
    char moviment[MAXPDSTRING];
    switch (type) {
    case 0: {
        pd_snprintf(moviment, MAXPDSTRING, "mouse_down");
    }
    case 1: {
        pd_snprintf(moviment, MAXPDSTRING, "mouse_up");
    }
    case 2: {
        pd_snprintf(moviment, MAXPDSTRING, "mouse_move");
    }
    case 3: {
        pd_snprintf(moviment, MAXPDSTRING, "mouse_drag");
    }
    }
    // TODO: run _mouseevent
}

// ─────────────────────────────────────
static void pdpy_gfx_mouse_down(t_pdpy_pdobj *o, int x, int y) { pdpy_gfx_mouse_event(o, x, y, 0); }
static void pdpy_gfx_mouse_up(t_pdpy_pdobj *o, int x, int y) { pdpy_gfx_mouse_event(o, x, y, 1); }
static void pdpy_gfx_mouse_move(t_pdpy_pdobj *o, int x, int y) { pdpy_gfx_mouse_event(o, x, y, 2); }
static void pdpy_gfx_mouse_drag(t_pdpy_pdobj *o, int x, int y) { pdpy_gfx_mouse_event(o, x, y, 3); }

// ─────────────────────────────────────
void pdpy_gfx_repaint(t_pdpy_pdobj *o, int firsttime) {
    // o->first_draw = firsttime;
    // // TODO: call repaint
    // o->first_draw = false;
    post("done");
}

// ─────────────────────────────────────
static void pdpy_gfx_clear(t_pdpy_pdobj *obj, int layer, int removed) {
    t_canvas *cnv = glist_getcanvas(obj->canvas);
    // TODO:

    // if (layer < obj->num_layers) {
    //     pdgui_vmess(0, "crs", cnv, "delete",
    //                 layer == -1 ? obj->object_tag : obj->layer_tags[layer]);
    // }

    // if (removed && object_tag->order_tag[0] != '\0') {
    //     pdgui_vmess(0, "crs", cnv, "delete", obj->order_tag);
    //     obj->order_tag[0] = '\0';
    // }
    //
    // glist_eraseiofor(obj->canvas, &obj->obj, obj->object_tag);
}

// ─────────────────────────────────────
static void pdpy_motion(t_gobj *z, t_floatarg dx, t_floatarg dy, t_floatarg up) {
    if (!up) {
        t_pdpy_pdobj *x = (t_pdpy_pdobj *)z;
        x->mouse_drag_x = x->mouse_drag_x + dx;
        x->mouse_drag_y = x->mouse_drag_y + dy;
        int zoom = glist_getzoom(glist_getcanvas(x->canvas));
        int xpos = (x->mouse_drag_x - text_xpix(&x->obj, x->canvas)) / zoom;
        int ypos = (x->mouse_drag_y - text_ypix(&x->obj, x->canvas)) / zoom;
        pdpy_gfx_mouse_drag(x, xpos, ypos);
    }
}

// ╭─────────────────────────────────────╮
// │           WIDGET BEHAVIOR           │
// ╰─────────────────────────────────────╯
void pdpy_getrect(t_gobj *z, t_glist *glist, int *xp1, int *yp1, int *xp2, int *yp2) {
    t_pdpy_pdobj *x = (t_pdpy_pdobj *)z;
    if (x->has_gui) {
        int zoom = glist->gl_zoom;
        float x1 = text_xpix((t_text *)x, glist), y1 = text_ypix((t_text *)x, glist);
        *xp1 = x1;
        *yp1 = y1;
        *xp2 = x1 + x->width * zoom;
        *yp2 = y1 + x->height * zoom;
    } else {
        // Bypass to text widgetbehaviour if we're not a GUI
        text_widgetbehavior.w_getrectfn(z, glist, xp1, yp1, xp2, yp2);
    }
}

// ─────────────────────────────────────
void pdpy_displace(t_gobj *z, t_glist *glist, int dx, int dy) {
    t_pdpy_pdobj *x = (t_pdpy_pdobj *)z;
    if (x->has_gui) {
        x->obj.te_xpix += dx, x->obj.te_ypix += dy;
        // dx *= glist_getzoom(glist);
        // dy *= glist_getzoom(glist);
    } else {
        text_widgetbehavior.w_displacefn(z, glist, dx, dy);
    }
    canvas_fixlinesfor(glist, (t_text *)x);
}

// ─────────────────────────────────────
void pdpy_delete(t_gobj *z, t_glist *glist) {
    if (!((t_pdpy_pdobj *)z)->has_gui) {
        text_widgetbehavior.w_deletefn(z, glist);
        return;
    }
    if (glist_isvisible(glist) && gobj_shouldvis(z, glist)) {
        pdpy_vis(z, glist, 0);
    }
    canvas_deletelinesfor(glist, (t_text *)z);
}

// ─────────────────────────────────────
int pdpy_click(t_gobj *z, t_glist *gl, int xpos, int ypos, int shift, int alt, int dbl, int doit) {
    t_pdpy_pdobj *x = (t_pdpy_pdobj *)z;
    // NOTE: Not run on plugdata

    if (x->has_gui) {
        int zoom = glist_getzoom(gl);
        int xpix = (xpos - text_xpix(&x->obj, gl)) / zoom;
        int ypix = (ypos - text_ypix(&x->obj, gl)) / zoom;

        if (doit) {
            if (!x->mouse_down) {
                pdpy_gfx_mouse_down(x, xpix, ypix);
                x->mouse_drag_x = xpos;
                x->mouse_drag_y = ypos;
            }
            glist_grab(x->canvas, &x->obj.te_g, (t_glistmotionfn)pdpy_motion, NULL, xpos, ypos);
        } else {
            pdpy_gfx_mouse_move(x, xpix, ypix);
            if (x->mouse_down) {
                pdpy_gfx_mouse_up(x, xpix, ypix);
            }
        }
        x->mouse_down = doit;
        return 1;
    } else {
        return text_widgetbehavior.w_clickfn(z, gl, xpos, ypos, shift, alt, dbl, doit);
    }
}

// ─────────────────────────────────────
void pdpy_vis(t_gobj *z, t_glist *glist, int vis) {
    t_pdpy_pdobj *x = (t_pdpy_pdobj *)z;
    if (!x->has_gui) {
        text_widgetbehavior.w_visfn(z, glist, vis);
        return;
    }

    // Otherwise, repaint or clear the custom graphics
    if (vis) {
        pdpy_gfx_repaint(x, 1);
    } else {
        pdpy_gfx_clear(x, -1, 1);
    }
}

// ─────────────────────────────────────
void pdpy_activate(t_gobj *z, t_glist *glist, int state) {
    if (!((t_pdpy_pdobj *)z)->has_gui) {
        text_widgetbehavior.w_activatefn(z, glist, state);
    }
}
