#include "fasteigenpy/info.hpp"
#include "fasteigenpy/cod.hpp"
#include "fasteigenpy/llt.hpp"
#include "fasteigenpy/ldlt.hpp"

// Eigen row-major matrix type
using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

PYBIND11_MODULE(_core, m) {
    wrap_info(m);

    wrap_cod<Eigen::MatrixXd>(m, "CompleteOrthogonalDecomposition");
    wrap_cod<MatrixXdR>(m, "CompleteOrthogonalDecompositionRowMajor");

    wrap_llt<Eigen::MatrixXd>(m, "LLT");
    wrap_llt<MatrixXdR>(m, "LLTRowMajor");

    wrap_ldlt<Eigen::MatrixXd>(m, "LDLT");
    wrap_ldlt<MatrixXdR>(m, "LDLTRowMajor");
}
