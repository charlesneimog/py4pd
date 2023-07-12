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
        #define PY4PD_EDITOR "idle3.10"
    #else
        #define PY4PD_EDITOR "idle3.10"
    #endif
#endif

// FOLDER
#include <dirent.h>

// ====================================
// =============== THREADS ============
// ====================================
/*
 * @brief Structure representing an outlet in the py4pd when it is detached.
 */
typedef struct {
    PyObject *pDict;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} py4pd_ThreadObjectInlet;

// ====================================
// ============== OUTLETS =============
// ====================================
/*
 * @brief Structure representing an array of all the auxiliar outlets of py4pd.
 */
typedef struct _py4pd_Outlets{
    t_atomtype u_type;
    t_outlet *u_outlet;
    int u_outletNumber;
} t_py4pd_Outlets;

// ====================================
// =============== PLAYER =============
// ====================================
/*
 * @brief Structure representing the values that are stored in the dictionary to be played for the player.
 */
typedef struct {
    int onset;
    int size;
    PyObject **values;
} KeyValuePair;

// ====================================
typedef struct {
    KeyValuePair* entries;
    int size;
    int lastOnset;
    int isSorted;
} Dictionary;


// ====================================
// ========== VIS OBJECTS =============
// ====================================

typedef struct _py4pd_edit_proxy{ 
    t_object    p_obj;
    t_symbol   *p_sym;
    t_clock    *p_clock;
    struct      _py *p_cnv;
}t_py4pd_edit_proxy;

// ====================================
// ============== PY4PD ===============
// ====================================

typedef struct _py { // It seems that all the objects are some kind of class.
    t_object             x_obj; // o objeto
    t_glist              *x_glist;
    t_py4pd_edit_proxy   *x_proxy; // para lidar com inlets auxiliares
    
    t_int                object_number; // object number
    t_int                runmode; // arguments
    t_int                visMode; // 1 for canvas, 2 for picture, 3 for score
    t_int                function_called; // flag to check if the set function was called
    t_int                py_arg_numbers; // number of arguments
    t_int                outPyPointer; // flag to check if is to output the python pointer
    t_int                kwargs;

    // Player 
    t_clock              *playerClock;
    Dictionary           *playerDict;
    t_int                  msOnset;
    t_int                  playerRunning;

    // Library
    t_int                py4pd_lib; // flag to check if is to use python library
    t_int                pyObject;
    t_int                ignoreOnNone;
    t_atom               *inlets; // vector to store the arguments
    PyObject             *argsDict; // parameters
    t_symbol             *objectName; // object name

    
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
    t_int               vectorSize; // vector size
    t_int               n_channels; // number of channels

    // == PICTURE AND SCORE
    t_int                 x_zoom; 
    t_int                 x_width;
    t_int                 x_height;
    t_int                 x_edit; // patch is in edit mode or not
    t_int                 x_init; // flag to check if the object was initialized
    t_int                 x_def_img; // flag to check if the object was initialized
    t_int                 x_sel; // flag to check if the object was selected
    t_int                 x_size;
    t_int                 x_latch;
    t_int                 x_numInlets;
    t_int                 x_numOutlets;
    t_int                 mouseIsOver;
    t_symbol            *x_fullname;
    t_symbol            *x_filename;
    t_symbol            *x_x;
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
    t_py4pd_Outlets     *outAUX; // outlets
    t_outlet            *out1; // outlet 1.
    t_inlet             *in1; // intlet 1
}t_py;

// =====================================
// =========== LIBRARY OBJECT ==========
// =====================================
typedef struct _py4pdInlet_proxy{
    t_object     p_ob;
    t_py        *p_master;
    int          inletIndex;
}t_py4pdInlet_proxy;

// =====================================
// =========== PYOBJECT IN PD ==========
// =====================================
/*
    * @brief this is used to store the pointer to PythonObject when we use PyObject in pd.
*/
typedef struct _pyObjectData {
    PyObject *pValue;
} t_pyObjectData;

// =====================================
extern void Py4pd_ReloadPy4pdFunction(t_py *x);
extern void Py4pd_SetParametersForFunction(t_py *x, t_symbol *s, int argc, t_atom *argv);
extern void Py4pd_PrintDocs(t_py *x);
extern void Py4pd_SetPythonPointersUsage(t_py *x, t_floatarg f);
extern void Py4pd_SetFunction(t_py *x, t_symbol *s, int argc, t_atom *argv);
extern void *Py4pd_ImportNumpyForPy4pd();
extern void *Py4pd_Py4pdFree(t_py *x);

#define PY4PD_IMAGE "R0lGODlhKgAhAPAAAP///wAAACH5BAAAAAAAIf8LSW1hZ2VNYWdpY2sOZ2FtbWE9MC40NTQ1NDUALAAAAAAqACEAAAIkhI+py+0Po5y02ouz3rz7D4biSJbmiabqyrbuC8fyTNf2jTMFADs="
#define PY4PDSIGTOTAL(s) ((t_int)((s)->s_length * (s)->s_nchans))

extern int pipePy4pdNum;
extern int object_count; 


// ============= METHODS ===============
void setEditor(t_py *x, t_symbol *s, int argc, t_atom *argv);

// ============= UTILITIES =============
int Py4pdUtils_ParseLibraryArguments(t_py *x, PyCodeObject *code, int argc, t_atom *argv);
t_py *Py4pdUtils_GetObject(void);
void Py4pdUtils_ParseArguments(t_py *x, t_canvas *c, int argc, t_atom *argv);
char* Py4pdUtils_GetFolderName(char* path);
const char* Py4pdUtils_GetFilename(const char* path);
void Py4pdUtils_CheckPkgNameConflict(t_py *x, char *folderToCheck, t_symbol *script_file_name);
void Py4pdUtils_FindObjFolder(t_py *x);
void Py4pdUtils_CreateTempFolder(t_py *x);
void Py4pdUtils_GetEditorCommand(t_py *x, char *command, int line);
void Py4pdUtils_ExecuteSystemCommand(const char *command);
int Py4pdUtils_IsNumericOrDot(const char *str);
void Py4pdUtils_RemoveChar(char *str, char c);
char *Py4pdUtils_Mtok(char *input, char *delimiter);
void Py4pdUtils_FromSymbolSymbol(t_py *x, t_symbol *s, t_outlet *outlet);
void *Py4pdUtils_PyObjectToPointer(PyObject *pValue);
PyObject *Py4pdUtils_PointerToPyObject(void *p);
void Py4pdUtils_FreePyObjectData(void *p);
PyObject *Py4pdUtils_RunPy(t_py *x, PyObject *pArgs);
void *Py4pdUtils_ConvertToPd(t_py *x, PyObject *pValue, t_outlet *outlet);
PyObject *Py4pdUtils_ConvertToPy(PyObject *listsArrays[], int argc, t_atom *argv);
void Py4pdUtils_SetObjConfig(t_py *x);
PyObject *Py4pdUtils_AddPdObject(t_py *x);
void Py4pdUtils_ReadGifFile(t_py *x, const char* filename);
void Py4pdUtils_ReadPngFile(t_py *x, const char* filename);
uint32_t Py4pdUtils_Ntohl(uint32_t netlong);
t_py *Py4pdUtils_GetObject(void);


// ============= EMBEDDED MODULE =======
extern PyMethodDef PdMethods[];
PyMODINIT_FUNC PyInit_pd(void);

// ============= PLAYER ==========

void PY4PD_Player_InsertThing(t_py *x, int onset, PyObject *value);
KeyValuePair* PY4PD_Player_GetValue(Dictionary* dictionary, int onset);

void py4pdPlay(t_py *x, t_symbol *s, int argc, t_atom *argv);
void py4pdStop(t_py *x);
void py4pdClear(t_py *x);

// ============= PIC =============
extern t_class *py4pd_class, *pyNewObject_VIS;
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
extern void *Py4pdLib_FreeObj(t_py *x);
extern PyObject *Py4pdLib_AddObj(PyObject *self, PyObject *args, PyObject *keywords);


// ===== TEST


#endif
