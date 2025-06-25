cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

pybind11_add_module(_core MODULE ${SOURCES} ${HEADERS})
target_include_directories(_core PRIVATE include)

#link with Eigen3
find_package(Eigen3 CONFIG REQUIRED)
target_include_directories(_core PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(_core PRIVATE Eigen3::Eigen)

# link with openmp
find_package(OpenMP REQUIRED)
target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)

# link with mkl
set(MKL_INTERFACE lp64) # need to use lp64 interface for MKL - NOT ilp64 which is incompatible with Eigen
find_package(MKL CONFIG)
#if we found MKL, we need to set the interface properties
if(MKL_FOUND)
    message(STATUS "Found MKL: ${MKL_VERSION}")
    target_compile_options(_core PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(_core PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(_core PUBLIC $<LINK_ONLY:MKL::MKL>)
    target_compile_definitions(_core PUBLIC EIGEN_USE_MKL_ALL=1) # use all MKL features
else()
    message(WARNING "MKL not found. Reverting to blas and lapack")
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    target_link_libraries(_core PRIVATE BLAS::BLAS LAPACK::LAPACK -llapacke)
    target_link_options(_core PRIVATE $<TARGET_PROPERTY:BLAS::BLAS_LINKER_FLAGS>)
    target_link_options(_core PRIVATE $<TARGET_PROPERTY:LAPACK::LAPACK_LINKER_FLAGS>)
    target_compile_definitions(_core PUBLIC EIGEN_USE_BLAS=1 EIGEN_USE_LAPACKE=1) # use blas and lapack
endif()

#specify the C++ standard
set_target_properties(_core PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# specify max compiler optimizations
target_compile_options(_core PRIVATE -O3 -march=native -mtune=native)

#specify eigen defines
target_compile_definitions(_core PRIVATE EIGEN_NO_DEBUG=1)


install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
