#ifndef FAST_EIGENPY_INFO_HPP
#define FAST_EIGENPY_INFO_HPP

#include <pybind11/pybind11.h>

#include <Eigen/Dense>

namespace py = pybind11;

void wrap_info(py::module& m) {
    py::enum_<Eigen::ComputationInfo>(m, "ComputationInfo")
        .value("Success", Eigen::Success)
        .value("NumericalIssue", Eigen::NumericalIssue)
        .value("NoConvergence", Eigen::NoConvergence)
        .value("InvalidInput", Eigen::InvalidInput)
        .export_values(); 
}

#endif
