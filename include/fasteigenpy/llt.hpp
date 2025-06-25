#ifndef FAST_EIGENPY_LLT_HPP
#define FAST_EIGENPY_LLT_HPP
/*
 * pybind11 freaks out if we try to explicitely wrap LLT
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
class WrappedLLT {
public:
    WrappedLLT() = default;
    WrappedLLT(const InputType& input)
        : llt(input) {}
    WrappedLLT(const Eigen::Index& rows)
        : llt(rows) {}

    void compute(const InputType& input) {
        llt.compute(input);
    }

    const InputType& matrixLLT() const {
        return llt.matrixLLT();
    }

    InputType matrixL() const {
        return llt.matrixL();
    }

    InputType matrixU() const {
        return llt.matrixL();
    }

    InputType reconstructedMatrix() const {
        return llt.reconstructedMatrix();
    }

    typename InputType::RealScalar rcond() const {
        return llt.rcond();
    }

    InputType solve(const Eigen::Ref<const Eigen::MatrixXd>& b) const {
        return llt.solve(b);
    }

    InputType transposeSolve(const Eigen::Ref<const Eigen::MatrixXd>& b) const {
        return llt.transpose().solve(b);
    }
    
    InputType adjointSolve(const Eigen::Ref<const Eigen::MatrixXd>& b) const {
        return llt.adjoint().solve(b);
    }

    //TODO: I think I definitely need to wrap this type
    Eigen::ComputationInfo info() const {
        return llt.info();
    }
private:
    Eigen::LLT<InputType> llt;
};

template <typename InputType>
void wrap_llt(py::module& m, const std::string& name){
    using THELLT = WrappedLLT<InputType>;
    py::class_<THELLT>(m, name.c_str())
        .def(py::init<>())
        .def(py::init<const Eigen::Ref<const InputType>&>())
        .def(py::init<const Eigen::Index&>(),
             py::arg("rows"))
        .def("info", &THELLT::info)
        .def("compute", &THELLT::compute,
             py::arg("input"))
        .def("matrixLLT", &THELLT::matrixLLT,
             py::return_value_policy::reference_internal)
        .def("matrixL", &THELLT::matrixL)
        .def("matrixU", &THELLT::matrixU)
        .def("reconstructedMatrix", &THELLT::reconstructedMatrix)
        .def("solve", &THELLT::solve)
        .def("transposeSolve", &THELLT::transposeSolve)
        .def("adjointSolve", &THELLT::adjointSolve);
}

#endif
