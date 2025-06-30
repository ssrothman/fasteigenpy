import eigenpy as eigen
import fasteigenpy as fasteigen

from time import time

import numpy as np

X = np.random.random((3000, 3000))

t0 = time()
cod = eigen.CompleteOrthogonalDecomposition(X)
cod.pseudoInverse()
print("Eigenpy (X) COD time:", time() - t0)
t0 = time()
cod = eigen.CompleteOrthogonalDecomposition(X.T)
cod.pseudoInverse()
print("Eigenpy (X.T) COD time:", time() - t0)

t0 = time()
cod = fasteigen.CompleteOrthogonalDecomposition(X)
cod.pseudoInverse()
print("Fast Eigenpy (X) COD time:", time() - t0)
t0 = time()
cod = fasteigen.CompleteOrthogonalDecomposition(X.T)
cod.pseudoInverse()
print("Fast Eigenpy (X.T) COD time:", time() - t0)

t0 = time()
cod = fasteigen.CompleteOrthogonalDecompositionRowMajor(X)
cod.pseudoInverse()
print("Fast RowMajor Eigenpy (X) COD time:", time() - t0)
t0 = time()
cod = fasteigen.CompleteOrthogonalDecompositionRowMajor(X.T)
cod.pseudoInverse()
print("Fast RowMajor Eigenpy (X.T) COD time:", time() - t0)

