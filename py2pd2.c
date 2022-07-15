#include "m_pd.h"
#include "g_canvas.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <windows.h>
#include <ShellApi.h>
#include <stdio.h>  

// ================================================================
// ================================================================
// ================================================================

static t_class *py_class;

// ============================================
typedef struct _py { // It seems that all the objects are some kind of class.
    t_object        x_obj; // convensao no puredata source
    t_symbol        *x_pyvariables; // array variables???
    t_symbol        *x_pyenvironment; // function name
    t_symbol        *x_pyscript; // symbol of the function
    t_float         *debug; // float value
    t_canvas        *x_canvas; // canvas
    t_outlet        *out_A; // outlet 1.
}t_py;

// ====================================
// Funcao inicializadora
// ====================================

void *py_new(void){
    t_py *x = (t_py *)pd_new(py_class); // pointer para a classe
    x->out_A = outlet_new(&x->x_obj, &s_anything); // cria um outlet
    return(x);
}

// =============================== Define enviroment ====================

static void py_env(t_py *x, t_symbol *name)
{   
    x->x_pyenvironment = name;
}

// =============================== Define enviroment ====================

// TODO - Make a Debug Function that change if will be SW_SHOW or SW_HIDE

static void py_debug(t_py *x, t_float *bool)
{   
    x->debug = bool;
}

// =============================== Define Script ====================

static void py_script(t_py *x, t_symbol *name)
{
    x->x_pyscript = NULL;
    x->x_pyscript = name;
}

// =============================== Add variables ====================

static void py_add_variable(t_py *x, t_symbol *s, int argc, t_atom *argv){
    s = NULL;

    if (argc > 2) {
        int i;
        char arg_list[1500];
        sprintf(arg_list, "%s", " "); // initialize the string
        for(i = 1; i < argc; i++)
        {
            if(argv[i].a_type == A_SYMBOL) // list symbols (PATHNAMES WITH SPACES)
            {
                t_symbol *sym = atom_gensym(argv+i);
                sprintf(arg_list, "%s %s", arg_list, sym->s_name);
            } 
            else if(argv[i].a_type == A_FLOAT) // list of numbers
            {
                if (atom_getfloat(argv+i) == roundf(atom_getfloat(argv+i))){
                    int num = roundf(atom_getfloat(argv+i));                   
                    sprintf(arg_list, "%s%d, ", arg_list, num);
                    }
                else { 
                    float num = atom_getfloat(argv+i);
                    sprintf(arg_list, "%s%f, ", arg_list, num);
                }
            }
        }
        
        t_symbol *var_name = atom_gensym(argv+0); // + 0 is necessary??????? 
        char final_string[500];

        sprintf(final_string, "%s = [%s] \n", var_name->s_name, arg_list);
        t_symbol *new_txt_variable = gensym(final_string);
        if (x->x_pyvariables == NULL) {
            x->x_pyvariables = new_txt_variable;
        } else {
            // x->x_pyvariables is equal to x_pyvariables + new_txt_variable
            sprintf(final_string, "%s%s", x->x_pyvariables->s_name, new_txt_variable->s_name);
            x->x_pyvariables = gensym(final_string);
        }

        // print x_pyvariables
        post("%s", final_string);
    }
    
    // ===============================================================
    // ===============================================================
    // ===============================================================
    
    else {
        // defining text of python variable
        t_symbol *var_name = atom_gensym(argv);
        char new_txt_variable_value[128];
        
        if  (argv[1].a_type == A_SYMBOL){
                t_symbol *sym = atom_gensym(argv+1);
                sprintf(new_txt_variable_value, "%s = %s \n #", var_name->s_name, sym->s_name);
            }
        else{
            if (atom_getfloat(argv+1) == roundf(atom_getfloat(argv+1))){
                sprintf(new_txt_variable_value, "%s = %d \n", var_name->s_name, (int)atom_getfloat(argv+1));
                }
            else { 
                sprintf(new_txt_variable_value, "%s = %f \n", var_name->s_name, atom_getfloat(argv+1));
            }
        }

        // add variables to the script
        t_symbol *new_txt_variable = gensym(new_txt_variable_value);
        if (x->x_pyvariables == NULL) {
            x->x_pyvariables = new_txt_variable;
        } else {
            // TODO = overwrite the variable if it already exists
            char set_of_variables[500];
            sprintf(set_of_variables, "%s %s", x->x_pyvariables->s_name, new_txt_variable->s_name);
            x->x_pyvariables = gensym(set_of_variables);
        }
        post("========================");
        post ("%s", x->x_pyvariables->s_name);
    }
}


// =============================== It defines the fundamental of the py ====================
static void py_run(t_py *x){    
    post("Running...");
    char script_path[128]; // get the path of the script
    sprintf(script_path, "%s", x->x_pyscript->s_name);  // get the name of the script 
    FILE* fh;
    char all_script[22000]; // TODO: How to know the size of the script?
    fopen_s(&fh, script_path, "r");
    if (fh == NULL){
        pd_error("file does not exists %s", script_path);
        return;
    }
    else{
        const size_t line_size = 300; // TODO: How to know the size of the LINES?
        char* line = malloc(line_size);
        while (fgets(line, line_size, fh) != NULL)  {
            
            strcat(all_script, line); // save lines in all_script
        }
        
        fclose(fh); // close file

        // add two lines at the BEGIN of the script
        char* script = malloc(sizeof(char) * (strlen(x->x_pyvariables->s_name) + strlen(all_script) + 1)); // Aqui ve quanto vai precisar alocar
        strcpy(script, x->x_pyvariables->s_name);
            
        strcat(script, all_script);
        FILE* fh2;
        char* tmp_script = strcat(script_path, "_TMP.py");
        fopen_s(&fh2, tmp_script, "w");
        fprintf(fh2, "%s", script);
        fclose(fh2);

        // clear variable all_script
        memset(all_script, 0, sizeof(all_script));
        // Execute the script
        char cmd[128];
        sprintf(cmd, "/c %s %s", x->x_pyenvironment->s_name, tmp_script);

        // execute cmd using ShellExecuteEx
        // create a new thread to execute the command
        SHELLEXECUTEINFO sei = {0};
        sei.cbSize = sizeof(sei);
        sei.fMask = SEE_MASK_NOCLOSEPROCESS;
        // sei.lpVerb = "open";
        sei.lpFile = "cmd.exe ";
        sei.lpParameters = cmd;
        if (x->debug) {
            sei.nShow = SW_SHOW;
        }
        else {
            sei.nShow = SW_HIDE;
        }
    
        ShellExecuteEx(&sei);
        WaitForSingleObject(sei.hProcess, INFINITE);
        CloseHandle(sei.hProcess);

        // get user's Folder and py_values.txt
        char user_folder[128];
        sprintf(user_folder, "%s", getenv("USERPROFILE"));
        sprintf(user_folder, "%s\\py_values.txt", user_folder);

        FILE* fh3;
        fopen_s(&fh3, user_folder, "r");
        if (fh3 == NULL){
            pd_error(x, "Your script seems to get some error! Please check your script!");
            return;
        }
        else{
            
            // get size values of the lines in py_values.txt
            char* py_values_lines = malloc(sizeof(char) * 3000);
            while (fgets(py_values_lines, line_size, fh3) != NULL)  {
                outlet_symbol(x->out_A, gensym(py_values_lines));
            }
            fclose(fh3);

            // clear variable py_values_lines
            // outlet_free(x->out_A);

        }
        // remove the py_values.txt file
        remove(user_folder);
    }  
}

// ====================================
// Funcao de Finalizacao
// ====================================

void py2pd_free(t_py *x){
    outlet_free(x->out_A);
    pd_free((t_pd *)x);
}

// ====================================
// Chama a inicializacao da classe
// ====================================

void py2pd_setup(void){
    py_class =     class_new(gensym("py2pd"), // cria o objeto quando escrevemos py2pd
                        (t_newmethod)py_new, // o methodo de inicializacao | pointer genérico
                        (t_method)py2pd_free, // quando voce deleta o objeto
                        sizeof(t_py), // quanta memoria precisamos para esse objeto
                        CLASS_DEFAULT, // nao há uma GUI especial para esse objeto
                        0); // todos os outros argumentos por exemplo um numero seria A_DEFFLOAT
    
    // class_addmethod(py_class, (t_method)pycontrol_dir, gensym("path"), A_DEFFLOAT, A_DEFSYMBOL, 0);
    class_addmethod(py_class, (t_method)py_run, gensym("bang"), 0); // o methodo de bang
    class_addmethod(py_class, (t_method)py_script, gensym("script"), A_SYMBOL, 0); 
    class_addmethod(py_class, (t_method)py_debug, gensym("debug"), A_FLOAT, 0);
    class_addmethod(py_class, (t_method)py_add_variable, gensym("var"), A_GIMME, 0); 
    class_addmethod(py_class, (t_method)py_env, gensym("venv"), A_SYMBOL, 0); 
    class_addmethod(py_class, (t_method)py_run, gensym("run"), 0, 0); 
    }