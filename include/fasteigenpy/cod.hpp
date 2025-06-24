#ifndef FAST_EIGENPY_COD_HPP
#define FAST_EIGENPY_COD_HPP
/*
 * pybind11 freaks out if we try to explicitely wrap CompleteOrthogonalDecomposition
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
class WrappedCOD {
public:
    WrappedCOD() = default;
    WrappedCOD(const InputType& input)
        : cod(input) {}
    WrappedCOD(const Eigen::Index& rows, const Eigen::Index& cols)
        : cod(rows, cols) {}

    void compute(const InputType& input) {
        cod.compute(input);
    }

    const InputType& matrixQTZ() const {
        return cod.matrixQTZ();
    }

    const InputType& matrixT() const {
        return cod.matrixT();
    }

    InputType matrixZ() const {
        return cod.matrixZ();
    }

    typename InputType::RealScalar absDeterminant() const {
        return cod.absDeterminant();
    }

    typename InputType::RealScalar logAbsDeterminant() const {
        return cod.logAbsDeterminant();
    }

    Eigen::Index dimensionOfKernel() const {
        return cod.dimensionOfKernel();
    }

    Eigen::Index rank() const {
        return cod.rank();
    }

    bool isInjective() const {
        return cod.isInjective();
    }

    bool isInvertible() const {
        return cod.isInvertible();
    }

    bool isSurjective() const {
        return cod.isSurjective();
    }

    typename InputType::RealScalar maxPivot() const {
        return cod.maxPivot();
    }

    Eigen::Index nonzeroPivots() const {
        return cod.nonzeroPivots();
    }

    InputType solve(const Eigen::Ref<const Eigen::MatrixXd>& b) const {
        return cod.solve(b);
    }

    InputType transposeSolve(const Eigen::Ref<const Eigen::MatrixXd>& b) const {
        return cod.transpose().solve(b);
    }
    
    InputType adjointSolve(const Eigen::Ref<const Eigen::MatrixXd>& b) const {
        return cod.adjoint().solve(b);
    }

    Eigen::Inverse<Eigen::CompleteOrthogonalDecomposition<InputType>> pseudoInverse() const {
        return cod.pseudoInverse();
    }

    typename InputType::RealScalar threshold() const {
        return cod.threshold();
    }

    //TODO: what is Eigen::PermutationType? Do I need to wrap it?
    const typename Eigen::CompleteOrthogonalDecomposition<InputType>::PermutationType& colsPermutation() const {
        return cod.colsPermutation();
    }

    //TODO: what is HCoeffsType? Do I need to wrap it?
    const typename Eigen::CompleteOrthogonalDecomposition<InputType>::HCoeffsType& hCoeffs() const {
        return cod.hCoeffs();
    }

    const typename Eigen::CompleteOrthogonalDecomposition<InputType>::HCoeffsType& zCoeffs() const {
        return cod.zCoeffs();
    }

    //TODO: what is this type? Do I need to wrap it?
    //typename Eigen::CompleteOrthogonalDecomposition<InputType>::householderSequenceType householderQ() const {
    //    return cod.householderQ();
    //}       

    //TODO: I think I definitely need to wrap this type
    Eigen::ComputationInfo info() const {
        return cod.info();
    }
private:
    Eigen::CompleteOrthogonalDecomposition<InputType> cod;
};

template <typename InputType>
void wrap_cod(py::module& m, const std::string& name){
    using THECOD = WrappedCOD<InputType>;
    py::class_<THECOD>(m, name.c_str())
        .def(py::init<>())
        .def(py::init<const Eigen::Ref<const InputType>&>())
        .def(py::init<const Eigen::Index&, const Eigen::Index&>(),
             py::arg("rows"), py::arg("cols"))
        .def("matrixQTZ", &THECOD::matrixQTZ,
             py::return_value_policy::reference_internal)
        .def("matrixT", &THECOD::matrixT,
             py::return_value_policy::reference_internal)
        .def("matrixZ", &THECOD::matrixZ)
        .def("absDeterminant", &THECOD::absDeterminant)
        .def("logAbsDeterminant", &THECOD::logAbsDeterminant)
        .def("dimensionOfKernel", &THECOD::dimensionOfKernel)
        .def("rank", &THECOD::rank)
        .def("isInjective", &THECOD::isInjective)
        .def("isInvertible", &THECOD::isInvertible)
        .def("isSurjective", &THECOD::isSurjective)
        .def("maxPivot", &THECOD::maxPivot)
        .def("nonzeroPivots", &THECOD::nonzeroPivots)
        .def("solve", &THECOD::solve)
        .def("transposeSolve", &THECOD::transposeSolve)
        .def("adjointSolve", &THECOD::adjointSolve)
        .def("pseudoInverse", &THECOD::pseudoInverse)
        .def("threshold", &THECOD::threshold);

}

#endif
