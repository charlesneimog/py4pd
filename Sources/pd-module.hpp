#pragma once

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

class PdModule {
  public:
    static void PdPrint(std::string s);
};

PyMODINIT_FUNC PyInit_pd();
