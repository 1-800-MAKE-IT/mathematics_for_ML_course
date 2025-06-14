In this assessment, you shall use the knowledge you have gained in  changing basis to construct a matrix that performs a transformation that  is easy in a particular basis, but complicated in our starting basis. Namely we shall help Panda Bear determine what his reflection will look like in a mirror that he has placed at an angle.

Background
Panda Bear is confused. He is trying to work out how things should look when reflected in a mirror, but is getting the wrong results. In Bear's coordinates, the mirror lies along the first axis. But, as is the way with bears, his coordinate system is not orthonormal: so what he thinks is the direction perpendicular to the mirror isn't actually the direction the mirror reflects in. Help Bear write a code that will do his matrix calculations properly!

Instructions
In this assignment you will write a Python function that will produce a transformation matrix for reflecting vectors in an arbitrarily angled mirror.

Building on the last assingment, where you wrote a code to construct an orthonormal basis that spans a set of input vectors, here you will take a matrix which takes simple form in that basis, and transform it into our starting basis. Recall the from the last video, the process for this. 
T=ETE^(−1)

You will write a function that will construct this matrix. This assessment is not conceptually complicated, but will build and test your ability to express mathematical ideas in code. As such, your final code submission will be relatively short, but you will receive less structure on how to write it.

Matrices in Python
For this exercise, we shall make use of the @ operator again. Recall from the last exercise, we used this operator to take the dot product of vectors. In general the operator will combine vectors and/or matrices in the expected linear algebra way, i.e. it will be either the vector dot product, matrix multiplication, or matrix operation on a vector, depending on it's input. 

You may need to use some of the following functions:

inv(A)
transpose(A)
gsBasis(A)

These, respectively, take the inverse of a matrix, give the transpose of a matrix, and produce a matrix of orthonormal column vectors given a general matrix of column vectors - i.e. perform the Gram-Schmidt process. This exercise will require you to combine some of these functions.