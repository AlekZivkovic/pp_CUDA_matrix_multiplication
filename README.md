# <p align = "center"> Parallel programming H3 </p>


Using the PyCuda environment write a CUDA program for (matrix) matrix multiplication. It takes a CUDA kernel to satisfy the Blas interface for generalized matrix multiplication:


gemm (transa, transb, m, n, k, alpha, A, B):
     alpha * (opA (A) X opB (B))

where:

<p>trance and transb: char parameters that define the opA and opB operations, as follows:</p>
<p>'N': op (X) = X - (does not change matrix)</p>
<p>'T': op (X) = X <sup> T </sup> (transposes matrix)</p>
<p>m - number of rows of matrix A</p>
<p>n is the number of columns of the matrix A</p>
<p>k - dimension of the missing matrix B (depending on the trance and transb can be the number of rows or the number of columns)</p>
<p>alpha - scalar value (float) which multiplies all elements of the resulting matrix</p>
<p>A and B - pointers to matrices A and B</p>
<p>Transposition should be built into the multiplication process (in case it is necessary to transpose a matrix, it should not be implemented as a special step of transposition before multiplication, but by changing the index in the source matrices which are used to calculate the results).</p>

Checks that the matrices are of appropriate dimensions in order for the operation to be performed should be performed in python code, before allocating memory on the CUDA device.

1. Program that performs multiplications of small matrices (multiplication can be done using a single block of thread)
2. A program that multiplies larger matrices, using a larger number of CUDA blocks.
3. Accelerate the solution from item 2 by using shared memory (so that no blocks first drag part of the data into the shared memory, and then read everything from shared memory)
