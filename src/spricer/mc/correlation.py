"""Correlation matrices: validation, and the nearest valid matrix when one is not.

A matrix typed by hand or estimated on gappy histories is often not positive definite. The
engine refuses such a matrix with the size of the problem instead of failing inside a Cholesky
factorisation; :func:`nearest_correlation` gives the closest valid matrix in Frobenius norm
(alternating projections with Dykstra's correction, Higham 2002, *Computing the nearest
correlation matrix*), to be used knowingly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from spricer._typing import FloatArray

_SYMMETRY_TOL = 1e-10
_MIN_EIGENVALUE = 1e-8  # floor that keeps the corrected matrix strictly positive definite


def check_correlation(matrix: ArrayLike, n_assets: int) -> FloatArray:
    """Return ``matrix`` as an array after checking it is a valid correlation matrix.

    Raises
    ------
    ValueError
        If the shape is not ``(n_assets, n_assets)``, an entry is not finite or outside
        ``[-1, 1]``, the matrix is not symmetric, its diagonal is not one, or it is not
        positive definite (the message gives its smallest eigenvalue).
    """
    c = np.asarray(matrix, dtype=np.float64)
    if c.shape != (n_assets, n_assets):
        raise ValueError(f"the correlation matrix must have shape {(n_assets, n_assets)}")
    if not np.all(np.isfinite(c)) or np.any(np.abs(c) > 1.0 + _SYMMETRY_TOL):
        raise ValueError("correlations must be finite and between -1 and 1")
    if not np.allclose(c, c.T, rtol=0.0, atol=_SYMMETRY_TOL):
        raise ValueError("the correlation matrix must be symmetric")
    if not np.allclose(np.diag(c), 1.0, rtol=0.0, atol=_SYMMETRY_TOL):
        raise ValueError("the diagonal of the correlation matrix must be one")
    smallest = float(np.linalg.eigvalsh(c)[0])
    if smallest <= 0.0:
        raise ValueError(
            f"the correlation matrix is not positive definite (smallest eigenvalue "
            f"{smallest:.4f}); these correlations are inconsistent with each other. "
            "nearest_correlation() gives the closest valid matrix."
        )
    return c


def _project_psd(x: FloatArray) -> FloatArray:
    values, vectors = np.linalg.eigh(0.5 * (x + x.T))
    return np.asarray((vectors * np.maximum(values, _MIN_EIGENVALUE)) @ vectors.T)


def nearest_correlation(
    matrix: ArrayLike, *, tol: float = 1e-10, max_iterations: int = 1000
) -> FloatArray:
    """Closest positive definite correlation matrix to a symmetric ``matrix`` (Higham 2002).

    Alternates the projection on positive semi-definite matrices (with Dykstra's correction)
    and the projection on unit-diagonal matrices until they agree to ``tol``. Eigenvalues are
    floored at a tiny positive value and the result is rescaled to a unit diagonal, so it is
    strictly positive definite and can be factorised. A valid matrix comes back unchanged.
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)):
        raise ValueError("expected a finite square matrix")
    a = 0.5 * (a + a.T)
    y = a.copy()
    correction = np.zeros_like(a)
    for _ in range(max_iterations):
        r = y - correction
        x = _project_psd(r)
        correction = x - r
        y_next = x.copy()
        np.fill_diagonal(y_next, 1.0)
        done = np.linalg.norm(y_next - y, "fro") <= tol * max(1.0, np.linalg.norm(y, "fro"))
        y = y_next
        if done:
            break
    x = _project_psd(y)
    d = 1.0 / np.sqrt(np.diag(x))
    out = x * d[:, None] * d[None, :]
    np.fill_diagonal(out, 1.0)
    return np.asarray(0.5 * (out + out.T))
