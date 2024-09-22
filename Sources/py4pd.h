// clang-format off
#ifndef PY4PD_H
#define PY4PD_H

#include <m_pd.h>
#include <g_canvas.h>
#include <s_stuff.h> // get the search 
#include <dirent.h>
#include <pthread.h>


#ifdef _WIN32
    #include <windows.h>
#else
   
    #include <pthread.h>
#endif

#define PY_SSIZE_T_CLEAN // Remove deprecated functions
#include <Python.h>

#ifdef __linux__
    #define __USE_GNU
#endif

#define PY4PD_NORMALOBJ 0
#define PY4PD_VISOBJ 1
#define PY4PD_AUDIOINOBJ 2
#define PY4PD_AUDIOOUTOBJ 3
#define PY4PD_AUDIOOBJ 4

#define PY4PD_MAJOR_VERSION 0
#define PY4PD_MINOR_VERSION 8
#define PY4PD_MICRO_VERSION 6

#define PY4PD_GIT_ISSUES "https://github.com/charlesneimog/py4pd/issues"
#define PYTHON_REQUIRED_VERSION(major, minor) ((major < PY_MAJOR_VERSION) || (major == PY_MAJOR_VERSION && minor <= PY_MINOR_VERSION))

// DEFINE STANDARD IDE EDITOR
#ifndef PY4PD_EDITOR
    #ifdef _WIN32
        // Windows
        #define PY4PD_EDITOR "idle3.12"
    #elif defined(__APPLE__) || defined(__MACH__)
        // macOS
        #define PY4PD_EDITOR "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m idlelib.idle"
    #else
        #define PY4PD_EDITOR "idle3.12"
    #endif
#endif

#define DEBUG 1

#if DEBUG
#define LOG(message, ...) \
    do { \
        fprintf(stderr, "[%s:%d] " message "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (0)
#else
#define LOG(message, ...)
#endif


// PY4PD 
extern t_class *Py4pdObjClass, *Py4pdVisObjClass;

// OUTLETS 
/*
 * @brief Structure representing an array of all the auxiliar outlets of py4pd.
 */
typedef struct _py4pdExtraOuts{
    t_atomtype u_type;
    t_outlet *u_outlet;
    int outCount;
} py4pdExtraOuts;

// PLAYER 
/*
 * @brief Structure representing the values that are stored in the dictionary to be played for the player.
 */
typedef struct {
    int Onset;
    int Size;
    PyObject **pValues;
} KeyValuePair;

typedef struct {
    KeyValuePair* Entries;
    int Size;
    int LastOnset;
    int isSorted;
} Dictionary;

// pValues Objs 
typedef struct _py4pd_pValue{ 
    PyObject* pValue;
    int IspValue;
    t_symbol *ObjOwner;
    int ClearAfterUse;
    int AlreadyIncref;
    int PdOutCount;
}t_py4pd_pValue;

// Collect objects
typedef struct pdcollectItem{
    char*         Key;
    PyObject*     pList;
    PyObject*     pItem;
    int           WasCleaned;
    int           aCumulative;
    int           Id;
} pdcollectItem;

typedef struct pdcollectHash{
    pdcollectItem** Itens;
    int Size;
    int Count;
} pdcollectHash;

// VIS OBJECTS 
typedef struct _py4pd_edit_proxy{ 
    t_object    p_obj;
    t_symbol   *p_sym;
    t_clock    *p_clock;
    struct      _py *p_cnv;
}t_py4pd_edit_proxy;

// PY4PD 
typedef struct _py { // It seems that all the objects are some kind of class.
    t_object            obj; // o objeto
    t_glist             *Glist;
    t_canvas            *Canvas; // pointer to the canvas
    t_outlet            *MainOut; // outlet 1.
    t_inlet             *MainIn; // intlet 1

    // TESTING THINGS
    t_py4pd_pValue      **PyObjArgs; // Obj Variables Arguments 
    pdcollectHash       *PdCollect; // hash table to store the objects
    t_int               RecursiveCalls; // check the number of recursive calls
    t_int               StackLimit; // check the number of recursive calls
    t_clock             *RecursiveClock; // clock to check the number of recursive calls 
    PyObject            *RecursiveObject; // object to check the number of recursive calls
    t_atom              *PdObjArgs;
    t_int               ObjArgsCount;
     
    // ===========
    t_int                VisMode; // is vis object
    t_int                FuncCalled; // flag to check if the set function was called
    t_int                pArgsCount; // number of arguments
    t_int                OutPyPointer; // flag to check if is to output the python pointer
    t_int                UsepArgs;
    t_int                UsepKwargs;
    
    // Player 
    t_clock              *PlayerClock;
    Dictionary           *PlayerDict;
    t_int                 PlayerRunning;
    t_int                 MsOnset;
    t_int                 Playable;

    // Library
    t_int                   IsLib; // flag to check if is to use python library
    t_int                   PyObject;
    t_int                   ObjType;
    t_int                   IgnoreOnNone;
    t_symbol                *ObjName; // object name
    PyObject                *ArgsDict; // parameters
    struct Py4pdNewObject   *ObjClass; // object class

    // == PYTHON
    PyThreadState       *pSubInterpState;
    t_int               pSubInterpRunning; // number of arguments

    PyObject            *pModule; // script name
    PyObject            *pFunction; // function name
    // PyObject            *showFunction; // TODO: FUTURE function to show the function
    PyObject            *KwArgsDict; // arguments
    PyObject            *pObjVarDict;
    PyObject            *pArgTuple;
    t_symbol            *pFuncName; // function_name; // function name
    t_symbol            *pScriptName; // script name or pathname

    // == AUDIO AND NUMPY
    t_int               AudioError; // To avoid multiple error messages
    t_int               AudioOut; // flag to check if is to use audio output
    t_int               AudioIn; // flag to check if is to use audio input
    t_int               PipGlobalInstall;
    t_int               UseNumpy; // flag to check if is to use numpy array in audioInput
    t_int               NumpyImported; // flag to check if numpy was imported
    t_float             Py4pdAudio; // audio to fe used in CLASSMAINSIGIN
    t_int               VectorSize; // vector size
    t_int               nChs; // number of channels

    // Pic Object
    t_int               Zoom; // zoom of the patch
    t_int               Width; // width of image
    t_int               Height; // height of image
    t_int               Edit; // patch is in edit mode or not
    t_int               DefImg; // flag to check if the object was initialized
    t_int               nInlets; // number of inlets
    t_int               nOutlets; // number of outlets
    t_int               MouseIsOver; // flag to check if the mouse is over the object
    t_symbol            *PicFilePath;
    t_symbol            *CanvasName; // name of the tcl variable for x
    char                *ImageBase64; // image data in base64
    t_int               DrawIOlets; // flag to check if the inlets and outlets were created

    // Paths
    t_symbol            *PkgPath; // packages path, where the packages are located
    t_symbol            *CondaPath; // Conda path, where the conda site-packages are located
    t_symbol            *PdPatchPath; // where the patch is located
    t_symbol            *Py4pdPath; // where py4pd object is located
    t_symbol            *TempPath; // temp path located in ~/.py4pd/, always is deleted when py4pd is closed
    t_symbol            *LibraryFolder; // where the library is located

    // script_name; // script name or pathname
    t_symbol            *EditorName; // editor name
    t_symbol            *EditorCommand; // editor personalized command
    py4pdExtraOuts      *ExtrasOuts; // outlets
    t_py4pd_edit_proxy  *Proxy; // para lidar com inlets auxiliares

}t_py;

// LIBRARY OBJECT 
typedef struct _py4pdInlet_proxy{
    t_object     p_ob;
    t_py        *p_master;
    int          inletIndex;
}t_py4pdInlet_proxy;

typedef struct _classInlets{
    t_object     p_ob;
    t_py        *p_master;
    int          inletIndex;
}classInlets;


void Py4pd_Pip(t_py *x, t_symbol *s, int argc, t_atom *argv);
void Py4pd_PrintDocs(t_py *x);
void Py4pd_SetPythonPointersUsage(t_py *x, t_floatarg f);
void Py4pd_SetFunction(t_py *x, t_symbol *s, int argc, t_atom *argv);

#define PY4PD_IMAGE "R0lGODlhKgAhAPAAAP///wAAACH5BAAAAAAAIf8LSW1hZ2VNYWdpY2sOZ2FtbWE9MC40NTQ1NDUALAAAAAAqACEAAAIkhI+py+0Po5y02ouz3rz7D4biSJbmiabqyrbuC8fyTNf2jTMFADs="
#define PY4PDSIGTOTAL(s) ((t_int)((s)->s_length * (s)->s_nchans))

extern int objCount; 
extern PyTypeObject Py4pdNewObj_Type;

#endif
