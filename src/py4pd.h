// clang-format off
#ifndef PY4PD_H
#define PY4PD_H

#include <m_pd.h>
#include <g_canvas.h>
#include <s_stuff.h> // get the search paths
#include <pthread.h>
#include <dirent.h>

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

#ifdef _WIN32
    #include <windows.h>
    #include <limits.h>

#endif

// DEFINE STANDARD IDE EDITOR
#ifndef PY4PD_EDITOR
    #ifdef _WIN32
        // Windows
        #define PY4PD_EDITOR "idle3.11"
    #elif defined(__APPLE__) || defined(__MACH__)
        // macOS
        #define PY4PD_EDITOR "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m idlelib.idle"
    #else
        #define PY4PD_EDITOR "idle3.11"
    #endif
#endif


// PY4PD 
extern t_class *py4pd_class, *pyNewObject_VIS;

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

// pValues Objs 
typedef struct _py4pd_pValue{ 
    PyObject* pValue;
    int isPvalue;
    int objectsUsing; 
    t_symbol *objOwner;
    int clearAfterUse;
    int alreadyIncref;
    int pdout;
}t_py4pd_pValue;

// Collect objects
typedef struct pdcollectItem{
    char*         key;
    PyObject*     pList;
    PyObject*     pItem;
    int           wasCleaned;
    int           aCumulative;
    int           id;
} pdcollectItem;

typedef struct pdcollectHash{
    pdcollectItem** items;
    int size;
    int count;
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
    t_glist             *glist;
    t_canvas            *canvas; // pointer to the canvas
    t_outlet            *mainOut; // outlet 1.
    t_inlet             *mainIn; // intlet 1

    // TESTING THINGS
    t_py4pd_pValue      **pyObjArgs; // Obj Variables Arguments 
    pdcollectHash       *pdcollect; // hash table to store the objects
    t_int               recursiveCalls; // check the number of recursive calls
    t_int               stackLimit; // check the number of recursive calls
    t_clock             *recursiveClock; // clock to check the number of recursive calls 
    PyObject            *recursiveObject; // object to check the number of recursive calls
    t_atom              *pdObjArgs;
    t_int               objArgsCount;
     
    // ===========
    t_int                visMode; // is vis object
    t_int                funcCalled; // flag to check if the set function was called
    t_int                pArgsCount; // number of arguments
    t_int                outPyPointer; // flag to check if is to output the python pointer
    t_int                use_pArgs;
    t_int                use_pKwargs;
    
    // Player 
    t_clock              *playerClock;
    Dictionary           *playerDict;
    t_int                 playerRunning;
    t_int                 msOnset;
    t_int                 playable;

    // Library
    t_int                   isLib; // flag to check if is to use python library
    t_int                   pyObject;
    t_int                   objType;
    t_int                   ignoreOnNone;
    t_symbol                *objName; // object name
    PyObject                *argsDict; // parameters
    struct Py4pdNewObject   *objClass; // object class

    // == PYTHON
    PyThreadState       *pSubInterpState;
    t_int               pSubInterpRunning; // number of arguments

    PyObject            *pModule; // script name
    PyObject            *pFunction; // function name
    // PyObject            *showFunction; // TODO: FUTURE function to show the function
    PyObject            *kwargsDict; // arguments
    PyObject            *pObjVarDict;
    PyObject            *pArgTuple;
    t_symbol            *pFuncName; // function_name; // function name
    t_symbol            *pScriptName; // script name or pathname

    // == AUDIO AND NUMPY
    t_int               audioOutput; // flag to check if is to use audio output
    t_int               pipGlobalInstall;
    t_int               audioInput; // flag to check if is to use audio input
    t_int               useNumpyArray; // flag to check if is to use numpy array in audioInput
    t_int               numpyImported; // flag to check if numpy was imported
    t_float             py4pdAudio; // audio to fe used in CLASSMAINSIGIN
    t_int               vectorSize; // vector size
    t_int               nChs; // number of channels
    t_int               audioError; // To avoid multiple error messages

    // Pic Object
    t_int               zoom; // zoom of the patch
    t_int               width; // width of image
    t_int               height; // height of image
    t_int               edit; // patch is in edit mode or not
    t_int               defImg; // flag to check if the object was initialized
    t_int               numInlets; // number of inlets
    t_int               numOutlets; // number of outlets
    t_int               mouseIsOver; // flag to check if the mouse is over the object
    t_symbol            *picFilePath;
    t_symbol            *canvasName; // name of the tcl variable for x
    char                *imageBase64; // image data in base64
    t_int               drawIOlets; // flag to check if the inlets and outlets were created

    // Paths
    t_symbol            *pkgPath; // packages path, where the packages are located
    t_symbol            *condaPath; // Conda path, where the conda site-packages are located
    t_symbol            *pdPatchPath; // where the patch is located
    t_symbol            *py4pdPath; // where py4pd object is located
    t_symbol            *tempPath; // temp path located in ~/.py4pd/, always is deleted when py4pd is closed
    t_symbol            *libraryFolder; // where the library is located

    // script_name; // script name or pathname
    t_symbol            *editorName; // editor name
    t_symbol            *editorCommand; // editor personalized command
    py4pdExtraOuts      *extrasOuts; // outlets
    t_py4pd_edit_proxy  *x_proxy; // para lidar com inlets auxiliares

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
