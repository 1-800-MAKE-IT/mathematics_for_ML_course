Task Description:

When coding or solving data analysis problems, one problem that can  occur is if your code encounters a special matrix that isn't invertible,  or has an infinite number of eigenvectors, or similar. On other  occasions, for example where you are reducing dimensionality, that might  even be desirable!  So here you will write a code fragment that traps  for different types of special matrices before calling the python  inversion routine, and classifies the type of special case encountered.  As much as anything, this is to give  you a chance to give a Python coding exercise a  try out in order to build confidence before doing some longer examples  later.


Identifying special matrices
Instructions
In this assignment, you shall write a function that will test if a 4×4 matrix is singular, i.e. to determine if an inverse exists, before calculating it.

You shall use the method of converting a matrix to echelon form, and testing if this fails by leaving zeros that can’t be removed on the leading diagonal.

Don't worry if you've not coded before, a framework for the function has already been written. Look through the code, and you'll be instructed where to make changes. We'll do the first two rows, and you can use this as a guide to do the last two.

Matrices in Python
In the numpy package in Python, matrices are indexed using zero for the top-most column and left-most row. I.e., the matrix structure looks like this:

A[0, 0]  A[0, 1]  A[0, 2]  A[0, 3]
A[1, 0]  A[1, 1]  A[1, 2]  A[1, 3]
A[2, 0]  A[2, 1]  A[2, 2]  A[2, 3]
A[3, 0]  A[3, 1]  A[3, 2]  A[3, 3]
You can access the value of each element individually using,

A[n, m]
which will give the n'th row and m'th column (starting with zero). You can also access a whole row at a time using,

A[n]
Which you will see will be useful when calculating linear combinations of rows.

A final note - Python is sensitive to indentation. All the code you should complete will be at the same level of indentation as the instruction comment.

How to submit
Edit the code in the cell below to complete the assignment. Once you are finished and happy with it, press the Submit Assignment button at the top of this notebook.

Please don't change any of the function names, as these will be checked by the grading script.

If you have further questions about submissions or programming assignments, here is a list of Q&A. You can also raise an issue on the discussion forum. Good luck!


Credit for task: Imperial College London