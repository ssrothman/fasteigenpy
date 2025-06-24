#include "fasteigenpy/cod.hpp"

// Eigen row-major matrix type
using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

PYBIND11_MODULE(_core, m) {
    wrap_cod<Eigen::MatrixXd>(m, "CompleteOrthogonalDecomposition");
    wrap_cod<MatrixXdR>(m, "CompleteOrthogonalDecompositionRowMajor");
}
