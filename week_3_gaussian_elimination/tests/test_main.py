import pytest
import numpy as np
from week_3_gaussian_elimination.main import isSingular, fixRowZero, fixRowOne, fixRowTwo, fixRowThree


@pytest.fixture
def matrix_B():
    """
    This matrix should NOT be singular, as it has a clear diagonal pivot.
    """
    return np.array([
        [2, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 4, 4],
        [0, 0, 5, 5]
    ], dtype=float)

@pytest.fixture
def matrix_A():
    """
    We originally thought A was singular, but the row-fixing code can pivot it.
    So in practice it ends up non-singular by the time each fixRow* is applied.
    """
    return np.array([
        [0, 7, -5, 3],
        [2, 8,  0, 4],
        [3,12,  0, 5],
        [1, 3,  1, 3]
    ], dtype=float)


def test_fixRowZero(matrix_B):
    # Ensure row0 pivot becomes 1
    B0 = fixRowZero(matrix_B.copy())
    assert B0[0,0] == pytest.approx(1.0)


def test_fixRowOne(matrix_A):
    # We fix row0 first, then row1
    A0 = fixRowZero(matrix_A.copy())
    A1 = fixRowOne(A0)
    # Now the sub-diagonal of row1 is 0, pivot is 1
    assert A1[1,0] == pytest.approx(0.0)
    assert A1[1,1] == pytest.approx(1.0)


def test_fixRowTwo(matrix_A):
    # Fix rows 0 and 1 first, then row2
    A0 = fixRowZero(matrix_A.copy())
    A1 = fixRowOne(A0)
    A2 = fixRowTwo(A1)
    # Row2’s sub-diagonals are zero, pivot is now 1
    assert A2[2,0] == pytest.approx(0.0)
    assert A2[2,1] == pytest.approx(0.0)
    assert A2[2,2] == pytest.approx(1.0)


def test_fixRowThree(matrix_A):
    # Fix rows 0-2 first, then row3
    A0 = fixRowZero(matrix_A.copy())
    A1 = fixRowOne(A0)
    A2 = fixRowTwo(A1)
    A3 = fixRowThree(A2)
    # Row3’s sub-diagonals are zero, pivot is now 1
    assert A3[3,0] == pytest.approx(0.0)
    assert A3[3,1] == pytest.approx(0.0)
    assert A3[3,2] == pytest.approx(0.0)
    assert A3[3,3] == pytest.approx(1.0)


def test_isSingular(matrix_A, matrix_B):
    # After applying fixRow*, A is actually invertible, so it is not singular
    assert not isSingular(matrix_A)
    # B was singular
    assert isSingular(matrix_B)