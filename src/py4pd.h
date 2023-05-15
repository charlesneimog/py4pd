#ifndef PY4PD_H
#define PY4PD_H

#include <m_pd.h>
#include <g_canvas.h>
#include <s_stuff.h>
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
extern void reload(t_py *x);
extern void set_param(t_py *x, t_symbol *s, int argc, t_atom *argv);
extern void documentation(t_py *x);
extern void usepointers(t_py *x, t_floatarg f);
extern void set_function(t_py *x, t_symbol *s, int argc, t_atom *argv);

#define PY4PD_IMAGE "R0lGODlh+gD6AOfdAAAAAAEBAQICAgMDAwQEBAUFBQYGBgcHBwgICAkJCQoKCgwMDA0NDQ4ODg8PDxAQEBERERISEhMTExQUFBUVFRYWFhcXFxgYGBkZGRoaGhsbGxwcHB0dHR4eHiAgICEhISMjIyQkJCUlJSYmJicnJygoKCkpKSsrKy0tLS4uLi8vLzAwMDExMTIyMjMzMzY2Njc3Nzg4ODk5OTo6Ojs7Ozw8PD09PT4+Pj8/P0BAQEFBQUJCQkNDQ0REREVFRUZGRkdHR0hISElJSUpKSktLS0xMTE1NTU9PT1BQUFFRUVJSUlNTU1RUVFVVVVZWVldXV1hYWFlZWVpaWltbW1xcXF5eXl9fX2FhYWJiYmNjY2RkZGVlZWZmZmdnZ2hoaGlpaWpqamtra2xsbG1tbW5ubm9vb3BwcHJycnNzc3V1dXd3d3h4eHl5eXt7e3x8fH9/f4CAgIGBgYKCgoODg4WFhYaGhoeHh4iIiImJiYqKioyMjI6Ojo+Pj5OTk5SUlJWVlZaWlpeXl5iYmJmZmZycnJ2dnZ6enp+fn6CgoKGhoaKioqOjo6WlpaampqioqKmpqaqqqqurq6ysrK2tra+vr7CwsLGxsbOzs7S0tLW1tba2tre3t7i4uLm5ubq6ury8vL29vb6+vr+/v8DAwMLCwsPDw8TExMXFxcbGxsfHx8jIyMnJycrKysvLy8zMzM3Nzc/Pz9DQ0NHR0dLS0tPT09TU1NXV1dbW1tfX19jY2NnZ2dra2tvb29zc3N3d3d/f3+Hh4eLi4uPj4+Tk5OXl5ebm5ufn5+jo6Onp6erq6uvr6+3t7e7u7u/v7/Dw8PHx8fLy8vPz8/T09PX19fb29vf39/j4+Pn5+fr6+vv7+/z8/P39/f7+/v///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////yH+EUNyZWF0ZWQgd2l0aCBHSU1QACwAAAAA+gD6AAAI/gABCBxIsKDBgwgTKlzIsKHDhxAjSpxIsaLFixgzatzIsaPHjwcdiBxJsqTJkyhTqlzJsqXLlzBjypxJs2bLgg666dzJs6fPn0CDCh1KtKjRo0iTKl3KtGlRBwRzOp1KtarVq1izag0KdaDUrWDDih1LtizPrgK/ml3Ltq3btmgBqH1Lt67du0fjzsXLt69fuFH/Ch5MWKvewogTKzZ6eLHjx4sbQ55Mua/kypgzs72subNnw4E/ix5NlTPp06iHmk7NunW31a5jj4Ytu7Zm2rZzT8atu7di3r6DDwYuvDhf4saT00WuvPla5s6ji4UuvXpW6tazlw6tvftY7N7D/uflLr68VfDm0/tEr779a/Lu4z+FL78+UPb2w+PP330//+z+/VddgAJGR2CBzR2IYHIKLlhcgw4GB2GEvU1IYW4WXlhbhhrGxmGHrX0IYmoijnhaiSbORl+K/a3IIoAuvjhgjDIaSGONCd6II4M67vhgjz5KCGSQFQ5JJIZGHrlhkkp6yGSTIT4JJYlSTnlilVaq6FWWPG7J5Y9efilkmGIWSWaZSJ6J5pJqrulkm25GCWecVM5J55V23qllWnrKhmKfy2EJ6G+CDorYn4ZuVmiiwy3K6F+IPkpWpJJO52ileFGK6VaabnrdpZ4GmmeohI5K6qGgnqqoqao2ymqr/pCmCuukss5q6au2ZlprrpzuyuunuP4qKp/CJtZpsYz5iux2wS77nLLONnVstFxBS61S0167nrXajtdst2BlC65O4o5bLrjndpuutute2y6170Ybr7PzLlsvsvcWm6+w+/7aL6//5hqwrQPPWjCsB7easKoLn9owqQ+HGrGnE29aMaYXV5qxpBs/2jGjHycasqEjD1oyoCf3mbKeK9/ZMp0vxxmzmzOvWTOaN5eZs5g7f9kzlz9nGbSVQ09ZNJRHN5m0kksf2TSRTwcZtY9T71g1jlfXmLWMW7/YNYtfpxi2iWOPWDaIZ3eYtoZrX9g2hW9HGLeDcy9YN4J3F5i3xoB7/9c3f3/nF7h9g9dXuHyHx5e4e4u317h6j6cXuXmTl1e5eJfrx+24qm3OebXffn5V5t6R3mLooldlunarw4h66lO1bp3sM74Ou7Se374T7dLxbqPtumObe/C+O1d8jsAHj9TxyjHfJbHKl+W8cdODCX303w2ve/XCcT/m9diH5b1v45sJfvi9Jo++UOXr1n6a56+P1fu20c9m/PKfp/3tccll0/8ADKAAB0jAAhowJSBJoAIXyMAGOvCBEIygBCdIwYQEBAA7"
#define PY4PD_SCORE "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"


extern int pipePy4pdNum;




#endif
