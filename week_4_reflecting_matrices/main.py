# GRADED FUNCTION
import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14 # That's 1×10⁻¹⁴ = 0.00000000000001

def gsBasis4(A):
    """Perform the Gram-Schmidt procedure for 4 basis vectors."""
    verySmallNumber = 1e-14  # Threshold for determining linear independence
    
    B = np.array(A, dtype=np.float64)  # Copy A as a float64 array
    
    # Step 1: Process column 0
    # Normalize column 0 (make it a unit vector)
    B[:, 0] = B[:, 0] / la.norm(B[:, 0])
    
    # Step 2: Process column 1
    # Orthogonalize column 1 (remove overlap with column 0)
    B[:, 1] = B[:, 1] - B[:, 1] @ B[:, 0] * B[:, 0]

    # Normalize column 1 (make it a unit vector if it is nonzero)
    if la.norm(B[:, 1]) > verySmallNumber:
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
    else:
        B[:, 1] = np.zeros_like(B[:, 1])  # Set to zero if dependent
    
    # Step 3: Process column 2
    # Orthogonalize column 2 (remove overlap with columns 0 and 1)
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 0] * B[:, 0]
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 1] * B[:, 1]

    # Normalize column 2 (make it a unit vector if it is nonzero)
    if la.norm(B[:, 2]) > verySmallNumber:
        B[:, 2] = B[:, 2] / la.norm(B[:, 2])
    else:
        B[:, 2] = np.zeros_like(B[:, 2])  # Set to zero if dependent
    
    # Step 4: Process column 3
    # Orthogonalize column 3 (remove overlap with columns 0, 1, and 2)
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 0] * B[:, 0]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 1] * B[:, 1]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 2] * B[:, 2]

    # Normalize column 3 (make it a unit vector if it is nonzero)
    if la.norm(B[:, 3]) > verySmallNumber:
        B[:, 3] = B[:, 3] / la.norm(B[:, 3])
    else:
        B[:, 3] = np.zeros_like(B[:, 3])  # Set to zero if dependent
    
    return B

def gsBasis(A) :
    """The second part of this exercise will generalise the procedure.
    Previously, we could only have four vectors, and there was a lot of repeating in the code.
    "We'll use a for-loop here to iterate the process for each vector."""

    B = np.array(A, dtype=float) # Make B as a copy of A, since we're going to alter it's values.
    # Loop over all vectors, starting with zero, label them with i
    for i in range(B.shape[1]) :
        # Inside that loop, loop over all previous vectors, j, to subtract.
        for j in range(i) :
            # subtract the overlap with previous vectors.
            # current vector B[:, i] and a previous vector B[:, j]
            B[:, i] =  B[:, i] - B[:, i] @ B[:, j] * B[:, j]

        # normalisation test for B[:, i]
        if la.norm(B[:, i]) > verySmallNumber:
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else:
            B[:, i] = np.zeros_like(B[:, i])  # Set to zero if dependent
        
    # Finally, we return the result:
    return B

# This function uses the Gram-schmidt process to calculate the dimension
# spanned by a list of vectors.
# Since each vector is normalised to one, or is zero,
# the sum of all the norms will be the dimension.
def dimensions(A) :
    return np.sum(la.norm(gsBasis(A), axis=0))
