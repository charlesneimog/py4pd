#ifndef PY4PD_H
#define PY4PD_H

#include <m_pd.h>
#include <g_canvas.h>


// for thread
#include <fcntl.h>
// #include <sys/mman.h>
#include <signal.h>
#include <sys/wait.h>
// ===


// in _WIN64 include windows.h, if not, include <pthread.h>
#define PY_SSIZE_T_CLEAN // Good practice to use this before include Python.h because it will remove some deprecated function

#ifdef _WIN64 
    #include <windows.h>  // on Windows, system() open a console window and we don't want that
    #include <Python.h>
#else
    #include <pthread.h>
    #include <Python.h>
    // #include <dirent.h>
#endif

#define PY4PD_MAJOR_VERSION 0
#define PY4PD_MINOR_VERSION 6
#define PY4PD_MICRO_VERSION 1

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
    t_float             py4pd_audio; // audio
    t_int               object_number; // object number
    t_int               thread; // arguments
    t_int               threadRunning; // flag to check if the thread is running
    t_int               visMode; // picture mode
    t_int               function_called; // flag to check if the set function was called
    t_int               py_arg_numbers; // number of arguments

    // Library
    t_int                py4pd_lib; // flag to check if is to use python library
    t_atom               *inlets; // vector to store the arguments
    PyObject            *argsDict; // parameters

    
    // == PYTHON
    PyObject            *module; // script name
    PyObject            *function; // function name
    PyObject            *Dict; // arguments
    
    // == AUDIO AND NUMPY
    t_int               audioOutput; // flag to check if is to use audio output
    t_int               audioInput; // flag to check if is to use audio input
    t_int               use_NumpyArray; // flag to check if is to use numpy array in audioInput
    t_int               numpyImported; // flag to check if numpy was imported

    // == PICTURE AND SCORE
    int             x_zoom;
    int             x_width;
    int             x_height;
    int             x_snd_set;
    int             x_rcv_set;
    int             x_edit;
    int             x_init;
    int             x_def_img;
    int             x_sel;
    int             x_outline;
    int             x_s_flag;
    int             x_r_flag;
    int             x_flag;
    int             x_size;
    int             x_latch;
    t_symbol        *file_name_open;
    t_symbol        *x_fullname;
    t_symbol        *x_filename;
    t_symbol        *x_x;
    t_symbol        *x_receive;
    t_symbol        *x_rcv_raw;
    t_symbol        *x_send;
    t_symbol        *x_snd_raw;

    t_canvas        *x_canvas; // pointer to the canvas

    t_symbol        *editorName; // editor name

    // == PATHS
    t_symbol        *packages_path; // packages path
    t_symbol        *home_path; // home path this always is the path folder (?)
    t_symbol        *py4pd_folder; // save object path  
    t_symbol        *temp_folder;
    t_symbol        *py4pd_scripts;


    t_symbol        *function_name; // function name
    t_symbol        *script_name; // script name
    t_outlet        *out_A; // outlet 1.
    t_inlet         *in1; // intlet 1
}t_py;

// =====================================
typedef struct _py4pdInlet_proxy{
    t_object     p_ob;
    t_py        *p_master;
    int          inletIndex;
}t_py4pdInlet_proxy;

// =====================================
typedef struct outsFromFork{
    int            outSize;
    t_atom          *out;
    int             error;
}t_outsFromFork;

// // =====================================
// struct thread_arg_struct {
//     t_py x;
//     t_symbol s;
//     int argc;
//     t_atom *argv;
//     PyThreadState *interp;
// } thread_arg;
//


// =====================================

extern void set_function(t_py *x, t_symbol *s, int argc, t_atom *argv);


#define PICIMAGE "R0lGODlh+gD6AOfdAAAAAAEBAQICAgMDAwQEBAUFBQYGBgcHBwgICAkJCQoKCgwMDA0NDQ4ODg8PDxAQEBERERISEhMTExQUFBUVFRYWFhcXFxgYGBkZGRoaGhsbGxwcHB0dHR4eHiAgICEhISMjIyQkJCUlJSYmJicnJygoKCkpKSsrKy0tLS4uLi8vLzAwMDExMTIyMjMzMzY2Njc3Nzg4ODk5OTo6Ojs7Ozw8PD09PT4+Pj8/P0BAQEFBQUJCQkNDQ0REREVFRUZGRkdHR0hISElJSUpKSktLS0xMTE1NTU9PT1BQUFFRUVJSUlNTU1RUVFVVVVZWVldXV1hYWFlZWVpaWltbW1xcXF5eXl9fX2FhYWJiYmNjY2RkZGVlZWZmZmdnZ2hoaGlpaWpqamtra2xsbG1tbW5ubm9vb3BwcHJycnNzc3V1dXd3d3h4eHl5eXt7e3x8fH9/f4CAgIGBgYKCgoODg4WFhYaGhoeHh4iIiImJiYqKioyMjI6Ojo+Pj5OTk5SUlJWVlZaWlpeXl5iYmJmZmZycnJ2dnZ6enp+fn6CgoKGhoaKioqOjo6WlpaampqioqKmpqaqqqqurq6ysrK2tra+vr7CwsLGxsbOzs7S0tLW1tba2tre3t7i4uLm5ubq6ury8vL29vb6+vr+/v8DAwMLCwsPDw8TExMXFxcbGxsfHx8jIyMnJycrKysvLy8zMzM3Nzc/Pz9DQ0NHR0dLS0tPT09TU1NXV1dbW1tfX19jY2NnZ2dra2tvb29zc3N3d3d/f3+Hh4eLi4uPj4+Tk5OXl5ebm5ufn5+jo6Onp6erq6uvr6+3t7e7u7u/v7/Dw8PHx8fLy8vPz8/T09PX19fb29vf39/j4+Pn5+fr6+vv7+/z8/P39/f7+/v///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////yH+EUNyZWF0ZWQgd2l0aCBHSU1QACwAAAAA+gD6AAAI/gABCBxIsKDBgwgTKlzIsKHDhxAjSpxIsaLFixgzatzIsaPHjwcdiBxJsqTJkyhTqlzJsqXLlzBjypxJs2bLgg666dzJs6fPn0CDCh1KtKjRo0iTKl3KtGlRBwRzOp1KtarVq1izag0KdaDUrWDDih1LtizPrgK/ml3Ltq3btmgBqH1Lt67du0fjzsXLt69fuFH/Ch5MWKvewogTKzZ6eLHjx4sbQ55Mua/kypgzs72subNnw4E/ix5NlTPp06iHmk7NunW31a5jj4Ytu7Zm2rZzT8atu7di3r6DDwYuvDhf4saT00WuvPla5s6ji4UuvXpW6tazlw6tvftY7N7D/uflLr68VfDm0/tEr779a/Lu4z+FL78+UPb2w+PP330//+z+/VddgAJGR2CBzR2IYHIKLlhcgw4GB2GEvU1IYW4WXlhbhhrGxmGHrX0IYmoijnhaiSbORl+K/a3IIoAuvjhgjDIaSGONCd6II4M67vhgjz5KCGSQFQ5JJIZGHrlhkkp6yGSTIT4JJYlSTnlilVaq6FWWPG7J5Y9efilkmGIWSWaZSJ6J5pJqrulkm25GCWecVM5J55V23qllWnrKhmKfy2EJ6G+CDorYn4ZuVmiiwy3K6F+IPkpWpJJO52ileFGK6VaabnrdpZ4GmmeohI5K6qGgnqqoqao2ymqr/pCmCuukss5q6au2ZlprrpzuyuunuP4qKp/CJtZpsYz5iux2wS77nLLONnVstFxBS61S0167nrXajtdst2BlC65O4o5bLrjndpuutute2y6170Ybr7PzLlsvsvcWm6+w+/7aL6//5hqwrQPPWjCsB7easKoLn9owqQ+HGrGnE29aMaYXV5qxpBs/2jGjHycasqEjD1oyoCf3mbKeK9/ZMp0vxxmzmzOvWTOaN5eZs5g7f9kzlz9nGbSVQ09ZNJRHN5m0kksf2TSRTwcZtY9T71g1jlfXmLWMW7/YNYtfpxi2iWOPWDaIZ3eYtoZrX9g2hW9HGLeDcy9YN4J3F5i3xoB7/9c3f3/nF7h9g9dXuHyHx5e4e4u317h6j6cXuXmTl1e5eJfrx+24qm3OebXffn5V5t6R3mLooldlunarw4h66lO1bp3sM74Ou7Se374T7dLxbqPtumObe/C+O1d8jsAHj9TxyjHfJbHKl+W8cdODCX303w2ve/XCcT/m9diH5b1v45sJfvi9Jo++UOXr1n6a56+P1fu20c9m/PKfp/3tccll0/8ADKAAB0jAAhowJSBJoAIXyMAGOvCBEIygBCdIwYQEBAA7"

#endif
