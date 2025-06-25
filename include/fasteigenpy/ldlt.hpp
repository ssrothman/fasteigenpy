#ifndef FAST_EIGENPY_LDLT_HPP
#define FAST_EIGENPY_LDLT_HPP
/*
 * pybind11 freaks out if we try to explicitely wrap LDLT
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
class WrappedLDLT {
public:
    WrappedLDLT() = default;
    WrappedLDLT(const InputType& input)
        : ldlt(input) {}
    WrappedLDLT(const Eigen::Index& rows)
        : ldlt(rows) {}

    void compute(const InputType& input) {
        ldlt.compute(input);
    }

    bool isPositive() const {
        return ldlt.isPositive();
    }

    bool isNegative() const {
        return ldlt.isNegative();
    }

    const InputType& matrixLDLT() const {
        return ldlt.matrixLDLT();
    }

    InputType matrixL() const {
        return ldlt.matrixL();
    }

    InputType matrixU() const {
        return ldlt.matrixL();
    }

    Eigen::VectorXd vectorD() const {
        return ldlt.vectorD();
    }

    InputType reconstructedMatrix() const {
        return ldlt.reconstructedMatrix();
    }

    typename InputType::RealScalar rcond() const {
        return ldlt.rcond();
    }

    InputType solve(const Eigen::Ref<const Eigen::MatrixXd>& b) const {
        return ldlt.solve(b);
    }

    InputType transposeSolve(const Eigen::Ref<const Eigen::MatrixXd>& b) const {
        return ldlt.transpose().solve(b);
    }
    
    InputType adjointSolve(const Eigen::Ref<const Eigen::MatrixXd>& b) const {
        return ldlt.adjoint().solve(b);
    }

    Eigen::ComputationInfo info() const {
        return ldlt.info();
    }

    InputType matrixPL() const {
        InputType result = ldlt.matrixL();
        result = ldlt.transpositionsP().transpose() * result;
        return result;
    }

private:
    Eigen::LDLT<InputType> ldlt;
};

template <typename InputType>
void wrap_ldlt(py::module& m, const std::string& name){
    using THELDLT = WrappedLDLT<InputType>;
    py::class_<THELDLT>(m, name.c_str())
        .def(py::init<>())
        .def(py::init<const Eigen::Ref<const InputType>&>())
        .def(py::init<const Eigen::Index&>(),
             py::arg("rows"))
        .def("info", &THELDLT::info)
        .def("isPositive", &THELDLT::isPositive) 
        .def("isNegative", &THELDLT::isNegative)
        .def("compute", &THELDLT::compute,
             py::arg("input"))
        .def("matrixLDLT", &THELDLT::matrixLDLT,
             py::return_value_policy::reference_internal)
        .def("matrixL", &THELDLT::matrixL)
        .def("matrixU", &THELDLT::matrixU)
        .def("vectorD", &THELDLT::vectorD)
        .def("matrixPL", &THELDLT::matrixPL)
        .def("reconstructedMatrix", &THELDLT::reconstructedMatrix)
        .def("solve", &THELDLT::solve)
        .def("transposeSolve", &THELDLT::transposeSolve)
        .def("adjointSolve", &THELDLT::adjointSolve);
}

#endif
