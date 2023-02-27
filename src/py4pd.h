#ifndef PY4PD_H
#define PY4PD_H
#include <m_pd.h>
#include <g_canvas.h>
#include <pthread.h>
#ifdef _WIN64 
    #include <windows.h>  // on Windows, system() open a console window and we don't want that
#endif
#define PY_SSIZE_T_CLEAN // Good practice to use this before include Python.h because it will remove some deprecated function
#include <Python.h>


#ifndef IHEIGHT
// Purr Data doesn't have these, hopefully the vanilla values will work
#define IHEIGHT 3       /* height of an inlet in pixels */
#define OHEIGHT 3       /* height of an outlet in pixels */
#endif

// =====================================
typedef struct _edit_proxy{
    t_object    p_obj;
    t_symbol   *p_sym;
    t_clock    *p_clock;
    struct      _py *p_cnv;
}t_edit_proxy;


// =====================================
typedef struct _py { // It seems that all the objects are some kind of class.
    t_object            x_obj; // convensao no puredata source code
    t_glist             *x_glist;
    t_edit_proxy        *x_proxy;
    t_float             py4pd_audio; // audio
    PyObject            *module; // python object
    PyObject            *function; // function name
    t_int               object_number; // object number
    t_int               thread; // arguments
    t_int               pictureMode; // picture mode
    t_int               function_called; // flag to check if the set function was called
    t_int               py_arg_numbers; // number of arguments
    t_int               use_NumpyArray; // flag to check if is to use numpy array in audioInput
    t_int               audioOutput; // flag to check if is to use audio output

    // == PICTURE AND SCORE
     int            x_zoom;
     int            x_width;
     int            x_height;
     int            x_snd_set;
     int            x_rcv_set;
     int            x_edit;
     int            x_init;
     int            x_def_img;
     int            x_sel;
     int            x_outline;
     int            x_s_flag;
     int            x_r_flag;
     int            x_flag;
     int            x_size;
     int            x_latch;
     t_symbol      *x_fullname;
     t_symbol      *x_filename;
     t_symbol      *x_x;
     t_symbol      *x_receive;
     t_symbol      *x_rcv_raw;
     t_symbol      *x_send;
     t_symbol      *x_snd_raw;
     t_outlet      *x_outlet;
    
    t_canvas            *x_canvas; // pointer to the canvas
    t_inlet             *in1; // intlet 1
    t_symbol            *editorName; // editor name
    t_symbol            *packages_path; // packages path 
    t_symbol            *home_path; // home path this always is the path folder (?)
    t_symbol            *object_path; // save object path   TODO: want to save scripts inside this folder and make all accessible
    t_symbol            *function_name; // function name
    t_symbol            *script_name; // script name
    t_outlet            *out_A; // outlet 1.
}t_py;

// Set Debug mode
#define DEBUG 1

#endif
