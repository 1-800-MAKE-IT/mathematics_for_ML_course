# PACKAGE
# Run this cell first once to load the dependancies.
import numpy as np
from numpy.linalg import norm, inv
from numpy import transpose
from readonly.bearNecessities import *

def build_reflection_matrix(bearBasis) :
    """In this function, you will return the transformation matrix T,
    having built it out of an orthonormal basis set E that you create from Bear's Basis
    and a transformation matrix in the mirror's coordinates T subscript E (referred to as T)"""

    # The parameter bearBasis is a 2×2 matrix that is passed to the function.
    # call gsBasis function on bearBasis to get the mirror's orthonormal basis.
    E = gsBasis(bearBasis)

    # Write a matrix in component form that performs the mirror's reflection in the mirror's basis.
    # Recall, the mirror operates by negating the last component of a vector.
    # Replace a,b,c,d with appropriate values
    TE = np.array([[1, 0],
                   [0, -1]])
    
    # Combine the matrices E and TE to produce your transformation matrix.
    T = E @ TE @ inv(E)

    # Finally, we return the result. There is no need to change this line.
    return T
