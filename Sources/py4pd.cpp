#include <m_pd.h>

extern "C" {
#include <m_imp.h>
#include <s_stuff.h> // get the search paths
}

// #include <filesystem>

#include "pd-module.hpp"
#include "py4pd.hpp"
#include "utils.hpp"

namespace py = pybind11;

py::object py_stdout;
py::object py_stderr;
py::object py_stdout_buffer;
py::object py_stderr_buffer;

// ─────────────────────────────────────
static bool Py4pd_LibraryLoad(Py4pdLib *x) {
    py::gil_scoped_acquire acquire; // Acquire the GIL here
    std::string LibScriptPath = x->PatchPath + "/" + x->LibScript + ".py";
    // printf("%s\n", LibScriptPath.c_str());
    // if (fs::exists(LibScriptPath)) {
    //     try {
    //         py::module LibModule = py::module::import(x->LibScript.c_str());
    //         return true;
    //     } catch (const std::exception &e) {
    //         post("[py4pd] %s", e.what());
    //         return false;
    //     }
    // }
    //
    // bool LibFound = false;
    // int i = 0;
    // while (true) {
    //     const char *PdPathPointer = namelist_get(STUFF->st_searchpath, i);
    //     if (!PdPathPointer) {
    //         break;
    //     }
    //     std::string PdPath = PdPathPointer;
    //     std::string LibPath = PdPath + "/" + x->LibScript + "/" + x->LibScript + ".py";
    //     post("[py4pd] %s", LibPath.c_str());
    //
    //     if (fs::exists(LibPath)) {
    //         //     std::string LibFolder = PdPath + x->LibScript;
    //         //     Py4pdUtils_AddPath(x, LibFolder);
    //         //     try {
    //         //         py::module LibModule = py::module::import(x->LibScript.c_str());
    //         //         LibFound = true;
    //         //         return true;
    //         //     } catch (const std::exception &e) {
    //         //         post("[py4pd] %s", e.what());
    //         //         return false;
    //         //     }
    //         //     break;
    //     }
    //     i++;
    // }
    //
    return false;
}

// ─────────────────────────────────────
static void *NewPy4pd(t_symbol *s, int argc, t_atom *argv) {

    bool LibMode = false;
    std::string LibScript;
    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            t_symbol *Py4pdArgs = atom_getsymbolarg(i, argc, argv);
            if (Py4pdArgs == gensym("-library") || Py4pdArgs == gensym("-lib")) {
                LibMode = true;
                LibScript = atom_getsymbolarg(i + 1, argc, argv)->s_name;
            }
        }
    }

    if (LibMode) {
        Py4pdLib *x;
        x = (Py4pdLib *)pd_new(Py4pdLibClass);
        x->LibScript = LibScript;
        x->Canvas = canvas_getcurrent();
        x->PatchPath = canvas_getdir(x->Canvas)->s_name;
        py::module SysPkg = py::module::import("sys");

        return x;
    } else {
        return nullptr;
    }
}

// ─────────────────────────────────────
static void *FreePy4pd(Py4pd *x) {
    Py4pdObjCount--;
    if (Py4pdObjCount == 0) {
        // Py_Finalize();
    }

    return nullptr;
}

// ─────────────────────────────────────
extern "C" void py4pd_setup(void) {
    Py4pdObjClass = class_new(gensym("py4pd"), (t_newmethod)NewPy4pd, (t_method)FreePy4pd,
                              sizeof(Py4pdObj), CLASS_DEFAULT, A_GIMME, 0);

    Py4pdLibClass = class_new(gensym("py4pd"), (t_newmethod)NewPy4pd, (t_method)FreePy4pd,
                              sizeof(Py4pdLib), CLASS_NOINLET, A_GIMME, 0);
    py::scoped_interpreter interpreter{};
}
