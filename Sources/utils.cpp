#include "utils.hpp"

void Py4pdUtils_AddPath(Py4pd *x, std::string Path) {
    py::gil_scoped_acquire acquire;
    py::module PkgSys = py::module::import("sys");
    py::list SysPath = PkgSys.attr("path");
    SysPath.attr("append")(Path);
}
