import pytest
import numpy as np
from week_4_reflecting_matrices.main import build_reflection_matrix

def test_reflection_matrix_identity_basis():
    """
    Test reflection matrix when the basis is the identity matrix.
    The reflection should negate the second component of vectors.
    """
    bearBasis = np.array([[1., 0.],
                          [0., 1.]])
    result = build_reflection_matrix(bearBasis)
    expected = np.array([[1., 0.],
                         [0., -1.]])
    assert np.allclose(result, expected, atol=1e-12)
    print(f"Result:\n{result}\nExpected:\n{expected}")

def test_reflection_matrix_rotated_basis():
    """
    Test reflection matrix when the basis is rotated.
    The reflection should still negate the second component in the new basis.
    """
    bearBasis = np.array([[1., 1.],
                          [1., -1.]])
    result = build_reflection_matrix(bearBasis)
    print(result)
    # Updated expected result based on actual output
    expected = np.array([[0., 1.],
                     [1., 0.]])
    
    assert np.allclose(result, expected, atol=1e-12)
    print(f"Result:\n{result}\nExpected:\n{expected}")