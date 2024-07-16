#pragma once

#include <m_pd.h>

#include <pybind11/embed.h> // python interpreter
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // type conversion

#include "pd-module.hpp"

namespace py = pybind11;
static t_class *Py4pdObjClass;
static t_class *Py4pdLibClass;
static int Py4pdObjCount = 0;

class Py4pd {
  public:
    t_sample Sample;

    // Paths
    t_canvas *Canvas;

    std::string Py4pdPath;
    std::string PatchPath;

    t_outlet *xOutlet;
};

class Py4pdObj : public Py4pd {
  public:
    t_object xObj;
};

class Py4pdLib : public Py4pd {
  public:
    t_object xObj;
    std::string LibScript;
};
