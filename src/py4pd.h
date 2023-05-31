#ifndef PY4PD_H
#define PY4PD_H

#include <m_pd.h>
#include <g_canvas.h>
#include <s_stuff.h> // get the search paths
#include <pthread.h>

#define PY_SSIZE_T_CLEAN // Good practice to use this before include Python.h because it will remove some deprecated function
#include <Python.h>

#ifdef _WIN64 
    #include <windows.h>  // on Windows, system() open a console window and we don't want that
#else
    #include <fcntl.h>
#endif

#define PY4PD_MAJOR_VERSION 0
#define PY4PD_MINOR_VERSION 7
#define PY4PD_MICRO_VERSION 0

// DEFINE STANDARD IDE EDITOR
#ifndef PY4PD_EDITOR
    #ifdef _WIN64
        #define PY4PD_EDITOR "notepad"
    #else
        #define PY4PD_EDITOR "gedit"
    #endif
#endif

// =====================================
typedef struct _py4pd_edit_proxy{ 
    t_object    p_obj;
    t_symbol   *p_sym;
    t_clock    *p_clock;
    struct      _py *p_cnv;
}t_py4pd_edit_proxy;

// =====================================
typedef struct _py { // It seems that all the objects are some kind of class.
    t_object            x_obj; // convensao no puredata source code
    t_glist             *x_glist;
    t_py4pd_edit_proxy  *x_proxy;
    
    t_int               object_number; // object number
    t_int               runmode; // arguments
    t_int               visMode; // 1 for canvas, 2 for picture, 3 for score
    t_int               function_called; // flag to check if the set function was called
    t_int               py_arg_numbers; // number of arguments
    t_int               outPyPointer; // flag to check if is to output the python pointer

    // Library
    t_int                py4pd_lib; // flag to check if is to use python library
    t_int                pyObject;
    t_atom               *inlets; // vector to store the arguments
    PyObject             *argsDict; // parameters
    t_symbol            *objectName; // object name

    
    // == PYTHON
    PyObject            *module; // script name
    PyObject            *function; // function name
    PyObject            *showFunction; // function to show the function
    PyObject            *Dict; // arguments
    
    // == AUDIO AND NUMPY
    t_int               audioOutput; // flag to check if is to use audio output
    t_int               audioInput; // flag to check if is to use audio input
    t_int               use_NumpyArray; // flag to check if is to use numpy array in audioInput
    t_int               numpyImported; // flag to check if numpy was imported
    t_float             py4pdAudio; // audio

    // == PICTURE AND SCORE
    int                 x_zoom;
    int                 x_width;
    int                 x_height;
    int                 x_edit;
    int                 x_init;
    int                 x_def_img;
    int                 x_sel;
    int                 x_outline;
    int                 x_size;
    int                 x_latch;
    int                 x_numInlets;
    int                 x_numOutlets;
    int                 mouseIsOver;
    // int             
    t_symbol            *file_name_open;
    t_symbol            *x_fullname;
    t_symbol            *x_filename;
    t_symbol            *x_x;
    t_symbol            *x_receive;
    t_symbol            *x_rcv_raw;
    t_symbol            *x_send;
    t_symbol            *x_snd_raw;
   

    t_canvas            *x_canvas; // pointer to the canvas

    t_symbol            *editorName; // editor name

    // == PATHS
    t_symbol            *pkgPath; // packages path, where the packages are located
    t_symbol            *pdPatchFolder; // where the patch is located
    t_symbol            *py4pdPath; // where py4pd object is located
    t_symbol            *tempPath; // temp path located in ~/.py4pd/, always is deleted when py4pd is closed




    t_symbol            *function_name; // function name
    t_symbol            *script_name; // script name
    t_outlet            *out1; // outlet 1.
    t_inlet             *in1; // intlet 1
}t_py;

// =====================================
typedef struct _py4pdInlet_proxy{
    t_object     p_ob;
    t_py        *p_master;
    int          inletIndex;
}t_py4pdInlet_proxy;

// =====================================
typedef struct _pyObjectData {
    PyObject *pValue;
} t_pyObjectData;
// =====================================
typedef enum{
    PY4PD_FLOAT, // 1
    PY4PD_SYMBOL, // 2
} py4pd_atomtype;

typedef struct _py4pdatom{
    float floatvalue;
    const char *symbolvalue;
    int a_type;
} py4pd_atom;

typedef struct _pdArgs{
    int size;
    py4pd_atom *atoms;
} pd_args;

// =====================================
extern void reloadPy4pdFunction(t_py *x);
extern void setParametersForFunction(t_py *x, t_symbol *s, int argc, t_atom *argv);
extern void printDocs(t_py *x);
extern void setPythonPointersUsage(t_py *x, t_floatarg f);
extern void setFunction(t_py *x, t_symbol *s, int argc, t_atom *argv);
extern void *importNumpyForPy4pd();
extern void *py4pdFree(t_py *x);

#define PY4PD_IMAGE "R0lGODlhKgAhAPAAAP///wAAACH5BAAAAAAAIf8LSW1hZ2VNYWdpY2sOZ2FtbWE9MC40NTQ1NDUALAAAAAAqACEAAAIkhI+py+0Po5y02ouz3rz7D4biSJbmiabqyrbuC8fyTNf2jTMFADs="


extern int pipePy4pdNum;
extern int object_count; 



#endif
