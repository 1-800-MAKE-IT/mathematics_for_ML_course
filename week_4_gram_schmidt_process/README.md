The Gram-Schmidt process is a method for constructing an orthonormal  basis of a space that a set of given vectors span. It can also be used  to determine the dimension of that space, which may be different than  the dimension of the vectors themselves or the number of vectors provided to span the space.

Instructions
In this assignment you will write a function to perform the Gram-Schmidt procedure, which takes a list of vectors and forms an orthonormal basis from this set. As a corollary, the procedure allows us to determine the dimension of the space spanned by the basis vectors, which is equal to or less than the space which the vectors sit.

You'll start by completing a function for 4 basis vectors, before generalising to when an arbitrary number of vectors are given.

Again, a framework for the function has already been written. Look through the code, and you'll be instructed where to make changes. We'll do the first two rows, and you can use this as a guide to do the last two.

Matrices in Python
Remember the structure for matrices in numpy is,

A[0, 0]  A[0, 1]  A[0, 2]  A[0, 3]
A[1, 0]  A[1, 1]  A[1, 2]  A[1, 3]
A[2, 0]  A[2, 1]  A[2, 2]  A[2, 3]
A[3, 0]  A[3, 1]  A[3, 2]  A[3, 3]
You can access the value of each element individually using,

A[n, m]
You can also access a whole row at a time using,

A[n]
Building on last assignment, in this exercise you will need to select whole columns at a time. This can be done with,

A[:, m]
which will select the m'th column (starting at zero).

In this exercise, you will need to take the dot product between vectors. This can be done using the @ operator. To dot product vectors u and v, use the code,

u @ v
All the code you should complete will be at the same level of indentation as the instruction comment.
