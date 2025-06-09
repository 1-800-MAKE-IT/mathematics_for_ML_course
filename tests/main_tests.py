import pytest
import numpy as np
from main import isSingular, fixRowZero


@pytest.fixture
def matrix_B():
    return np.array([
        [2, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 4, 4],
        [0, 0, 5, 5]
    ], dtype=float)

@pytest.fixture
def matrix_A():
    return np.array([
        [0, 7, -5, 3],
        [2, 8,  0, 4],
        [3,12,  0, 5],
        [1, 3,  1, 3]
    ], dtype=float)


def test_fixRowOne(matrix_A):
    assert fixRowOne(matrix_A)

def test_fixRowTwo(matrix_A):
    assert fixRowTwo(matrix_A)

def test_fixRowThree(matrix_A):
    assert fixRowThree(matrix_A)