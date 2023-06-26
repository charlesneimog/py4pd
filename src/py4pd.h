#ifndef PY4PD_H
#define PY4PD_H

#include <m_pd.h>
#include <g_canvas.h>
#include <s_stuff.h> // get the search paths
#include <pthread.h>

#define PY_SSIZE_T_CLEAN // Remove deprecated functions
#include <Python.h>

#ifdef _WIN64 
    #include <windows.h>  
#else
    #include <fcntl.h> // For pipes, TODO: Remove this
#endif

#ifdef __linux__
    #define __USE_GNU
#endif

#include <dlfcn.h> // for RTLD_LAZY, RTLD_NOW, RTLD_GLOBAL (find where is the root of the object)

#define PY4PD_MAJOR_VERSION 0
#define PY4PD_MINOR_VERSION 8
#define PY4PD_MICRO_VERSION 0

// DEFINE STANDARD IDE EDITOR
#ifndef PY4PD_EDITOR
    #ifdef _WIN64
        #define PY4PD_EDITOR "notepad"
    #else
        #define PY4PD_EDITOR "gedit"
    #endif
#endif


// ================ PLAYER =============
typedef struct {
    int onset;
    int size;
    PyObject **values;
} KeyValuePair;

typedef struct {
    KeyValuePair* entries;
    int size;
    int lastOnset;
    int isSorted;
} Dictionary;

// ================ VIS Object ========= 
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
    int                 kwargs;

    // Player 
    t_clock             *playerClock;
    Dictionary          *playerDict;
    int                 msOnset;
    int                 playerRunning;

    // Library
    t_int                py4pd_lib; // flag to check if is to use python library
    t_int                pyObject;
    t_int                  ignoreOnNone;
    t_atom               *inlets; // vector to store the arguments
    PyObject             *argsDict; // parameters
    t_symbol            *objectName; // object name

    
    // == PYTHON
    PyObject            *module; // script name
    PyObject            *function; // function name
    PyObject            *showFunction; // function to show the function
    PyObject            *Dict; // arguments
    PyObject            *kwargsDict; // arguments
    
    // == AUDIO AND NUMPY
    t_int               audioOutput; // flag to check if is to use audio output
    t_int               audioInput; // flag to check if is to use audio input
    t_int               use_NumpyArray; // flag to check if is to use numpy array in audioInput
    t_int               numpyImported; // flag to check if numpy was imported
    t_float             py4pdAudio; // audio
    int                 vectorSize; // vector size

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
    char                *x_image;
   

    t_canvas            *x_canvas; // pointer to the canvas

    t_symbol            *editorName; // editor name

    // == PATHS
    t_symbol            *pkgPath; // packages path, where the packages are located
    t_symbol            *pdPatchFolder; // where the patch is located
    t_symbol            *py4pdPath; // where py4pd object is located
    t_symbol            *tempPath; // temp path located in ~/.py4pd/, always is deleted when py4pd is closed
    t_symbol            *libraryFolder; // where the library is located

    t_symbol            *function_name; // function name
    t_symbol            *script_name; // script name or pathname
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

// TODO: REMOVE py4pd_atomtype, py4pd_atom, pd_args

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


// ============= UTILITIES =============
int parseLibraryArguments(t_py *x, PyCodeObject *code, int argc, t_atom *argv);
void parsePy4pdArguments(t_py *x, t_canvas *c, int argc, t_atom *argv);
void findPy4pdFolder(t_py *x);
void createPy4pdTempFolder(t_py *x);
void setPy4pdConfig(t_py *x);
char *getEditorCommand(t_py *x);
void executeSystemCommand(const char *command);
int isNumericOrDot(const char *str);
void removeChar(char *str, char c);
// --------
t_py *get_py4pd_object(void);
// --------
char* get_folder_name(char* path);
const char* get_filename(const char* path);
// --------
void *py4pd_convert_to_pd(t_py *x, PyObject *pValue);
PyObject *py4pd_convert_to_py(PyObject *listsArrays[], int argc, t_atom *argv);
PyObject *py4pd_add_pd_object(t_py *x);
void *pyobject_to_pointer(PyObject *pValue);
PyObject *pointer_to_pyobject(void *p);
void free_pyobject_data(void *p);
// --------
void readGifFile(t_py *x, const char* filename);
void readPngFile(t_py *x, const char* filename);
// --------
void py4pd_fromsymbol_symbol(t_py *x, t_symbol *s);
uint32_t py4pd_ntohl(uint32_t netlong);

// ============= EMBEDDED MODULE =======
extern PyMethodDef PdMethods[];
PyMODINIT_FUNC PyInit_pd(void);
t_py *get_py4pd_object(void);

// ============= PLAYER ==========

void PY4PD_Player_InsertThing(t_py *x, int onset, PyObject *value);
KeyValuePair* PY4PD_Player_GetValue(Dictionary* dictionary, int onset);

void py4pdPlay(t_py *x, t_symbol *s, int argc, t_atom *argv);
void py4pdStop(t_py *x);
void py4pdClear(t_py *x);

// ============= PIC =============
extern t_class *py4pd_class, *py4pd_class_VIS, *pyNewObject_VIS;
extern void PY4PD_free(t_py *x);
extern void PY4PD_zoom(t_py *x, t_floatarg f);
extern void py4pd_InitVisMode(t_py *x, t_canvas *c, t_symbol *py4pdArgs, int index, int argc, t_atom *argv, t_class *obj_class);
extern void PY4PD_erase(t_py* x, struct _glist *glist); 
extern void PY4PD_draw(t_py* x, struct _glist *glist, t_floatarg vis);
extern const char* PY4PD_filepath(t_py *x, const char *filename);

// widget
extern void PY4PD_getrect(t_gobj *z, t_glist *glist, int *xp1, int *yp1, int *xp2, int *yp2);
extern void PY4PD_displace(t_gobj *z, t_glist *glist, int dx, int dy);
extern void PY4PD_select(t_gobj *z, t_glist *glist, int state);
extern void PY4PD_delete(t_gobj *z, t_glist *glist);

// ============= EXTERNAL LIBRARIES =============
extern void *py_newObject(t_symbol *s, int argc, t_atom *argv);
extern void *py_freeObject(t_py *x);
extern PyObject *pdAddPyObject(PyObject *self, PyObject *args, PyObject *keywords);


// ===== TEST


#endif
