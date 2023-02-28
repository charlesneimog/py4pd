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
    t_int               audioInput; // flag to check if is to use audio input

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
    t_symbol        *file_name_open;
     t_symbol       *x_fullname;
     t_symbol       *x_filename;
     t_symbol       *x_x;
     t_symbol       *x_receive;
     t_symbol       *x_rcv_raw;
     t_symbol       *x_send;
     t_symbol       *x_snd_raw;
     t_outlet       *x_outlet;
    
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

#define DEBUG 1
#define SCOREIMAGE "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"
#define PICIMAGE "R0lGODlh7wDRAPQSAAAAAA8PDxAQEBYWFhgYGB8fHzQ0NDs7O0JCQkRERGBgYGdnZ2hoaGxsbHBwcH5+fo2NjZKSkv///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAAAAAAAIf8LSW1hZ2VNYWdpY2sOZ2FtbWE9MC40NTQ1NDUALAAAAADvANEAAAX/ICCOZGmeaKqubOu+cCzPdH1Cdq7vfO//QBJOFCgaj8ikcslsOp/QqHRKrVqv2MBoGJB4v+CweEwum8/otHrNbrvf8LjkIeLK7/i8fs/v8+kAdn6DhIWGh4iAgoiMjY6Pj4pEkJSVlpdtkgBdmJ2en5CanKCkpaZ4oqeqq6xmqa2wsaevsrW2lrS3uruGuby/wKgiEZPBxsduvsjLzGHKzdDIz9HUv9PV2LbX2dyt293gpt/h5J7j5eiV5+nsjevt8IXv8fR/dcX1+YyAxJv6/4fmARy4RiDBg2YK3POHsKEcAwtHOZzIxiBFhxYvIsyokSDHjgA/gtQncmS9kibj/6FM2S5BRJYstQTCBxPkyprlbuIMp3Nnt54+swENWm0o0WhGjzZLqnRZg5dNG26hGXUgA6hVPWLNGnIrV5LDqH496XUsPaZmdaFNq60sW3Zr38YS4FZuOQR17fLMq/cn375C/wIuKngw0rAMDZOLq5gU48afHkPudKDw5GMELF8OJnmzOs2eeXUOHQk06VujT+8zrVoWP7GtjaWOLY81bVazb/vJrdvezMS9gfEOnmc48TvGj8dJrvwN8+aZbEPv9Hy6murW0WDP7ko6d0rbv48JL96Z9/LuzqNPpH59r/buCZF3P399ffT3y+cXv/97f+7/ZRegda8BF59j8B3Yx/+A0zEInYPNQaichMdRSJxLv0mk4CcyLbJhZAl+WFyIIiJHYonLIaYhip9lyCJ1J74YnYsy4hJjjWo8RSOOkExlII+MLHAjkN3tSORqRh4Z0JBKimFhcE/2FqVuU95WJW0DMNkkGHglueUgV8YWZmtjqlbmaWeSlmZoa3rW5maVefllH5nJOeceBa54J55azvnmZX9OFihkgzZWqGKHGqZAn1/6qOeeeTjA6JaJDpYnpGBO2mSlgHHal6d6gWqXqHKR+papbKGalqpmsTrWpZguqKmSrn5VK1e3ZpVrVbtG1WtTvyoV7FHDElVsUMf6lOxOy+LUbE3PwhQtS9OmVK09SdeOlK1Nsx65bUffahTuReNSVO5E52LULZGabJLFu/DGK++89NY7r49B5Kvvvvz2y+8Q/gYs8MAEEwxBCAA7"



#endif
