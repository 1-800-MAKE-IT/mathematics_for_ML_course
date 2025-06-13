import pytest
import numpy as np
from week_4_gram_schmidt_process.main import gsBasis4, gsBasis, dimensions

def test_repeatability_of_gsBasis4():
    V = np.array([[1., 0., 2., 6.],
                  [0., 1., 8., 2.],
                  [2., 8., 3., 1.],
                  [1., -6., 2., 3.]], dtype=float)

    # Once you've done Gram-Schmidt once,
    # doing it again should give you the same result. Test this:
    U = gsBasis4(V)

    # Try the general function too.
    assert np.allclose(gsBasis(V), np.array([[0.40824829, -0.1814885, 0.04982278, 0.89325973],
                                             [0., 0.1088931, 0.99349591, -0.03328918],
                                             [0.81649658, 0.50816781, -0.06462163, -0.26631346],
                                             [0.40824829, -0.83484711, 0.07942048, -0.36063281]], dtype=float))

def test_non_square_matrices_behavior():
    # See what happens for non-square matrices
    A = np.array([[3., 2., 3.],
                  [2., 5., -1.],
                  [2., 4., 8.],
                  [12., 2., 1.]], dtype=float)
    assert np.allclose(gsBasis(A), np.array([[0.23643312, 0.18771349, 0.22132104],
                                             [0.15762208, 0.74769023, -0.64395812],
                                             [0.15762208, 0.57790444, 0.72904263],
                                             [0.94573249, -0.26786082, -0.06951101]], dtype=float))

    assert int(dimensions(A)) == 3

def test_gsBasis_on_matrix_B():
    B = np.array([[6., 2., 1., 7., 5.],
                  [2., 8., 5., -4., 1.],
                  [1., -6., 3., 2., 8.]], dtype=float)
    assert np.allclose(gsBasis(B), np.array([[0.93704257, -0.12700832, -0.32530002, 0., 0.],
                                             [0.31234752, 0.72140727, 0.61807005, 0., 0.],
                                             [0.15617376, -0.6807646, 0.71566005, 0., 0.]], dtype=float))

def test_linear_combination_behavior():
    # Now let's see what happens when we have one vector that is a linear combination of the others.
    C = np.array([[1., 0., 2.],
                  [0., 1., -3.],
                  [1., 0., 2.]], dtype=float)

    assert np.allclose(gsBasis(C), np.array([[0.70710678, 0., 0.],
                                             [0., 1., 0.],
                                             [0.70710678, 0., 0.]], dtype=float))

    assert int(dimensions(C)) == 2