#ifndef FAST_EIGENPY_SelfAdjointEigenSolver_HPP
#define FAST_EIGENPY_SelfAdjointEigenSolver_HPP
/*
 * pybind11 freaks out if we try to explicitely wrap SelfAdjointEigenSolver
 * I think this is because it inherits from Eigen::Base but isn't a matrix
 * So it falls into a special case in their predefined bindings.
 *
 * Solution: a transparant wrapper?
 */
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
namespace py = pybind11;

template <typename InputType>
class WrappedSelfAdjointEigenSolver {
public:
    WrappedSelfAdjointEigenSolver() = default;
    WrappedSelfAdjointEigenSolver(const InputType& input)
        : selfadjoineigensolver(input) {}
    WrappedSelfAdjointEigenSolver(const Eigen::Index& rows)
        : selfadjoineigensolver(rows) {}

    void compute(const InputType& input) {
        selfadjoineigensolver.compute(input);
    }

    const Eigen::VectorXd& eigenvalues() const {
        return selfadjoineigensolver.eigenvalues();
    }
    
    const InputType& eigenvectors() const {
        return selfadjoineigensolver.eigenvectors();
    }

    Eigen::ComputationInfo info() const {
        return selfadjoineigensolver.info();
    }

    InputType operatorSqrt() const {
        return selfadjoineigensolver.operatorSqrt();
    }

    InputType operatorInverseSqrt() const {
        return selfadjoineigensolver.operatorInverseSqrt();
    }
private:
    Eigen::SelfAdjointEigenSolver<InputType> selfadjoineigensolver;
};

template <typename InputType>
void wrap_selfadjoineigensolver(py::module& m, const std::string& name){
    using THESelfAdjointEigenSolver = WrappedSelfAdjointEigenSolver<InputType>;
    py::class_<THESelfAdjointEigenSolver>(m, name.c_str())
        .def(py::init<>())
        .def(py::init<const Eigen::Ref<const InputType>&>())
        .def(py::init<const Eigen::Index&>(),
             py::arg("rows"))
        .def("info", &THESelfAdjointEigenSolver::info)
        .def("compute", &THESelfAdjointEigenSolver::compute,
             py::arg("input"))
        .def("eigenvalues", &THESelfAdjointEigenSolver::eigenvalues,
             py::return_value_policy::reference_internal)
        .def("eigenvectors", &THESelfAdjointEigenSolver::eigenvectors,
                py::return_value_policy::reference_internal)
        .def("operatorSqrt", &THESelfAdjointEigenSolver::operatorSqrt)
        .def("operatorInverseSqrt", &THESelfAdjointEigenSolver::operatorInverseSqrt);
}

#endif
