#include "taco.h"
#include <iostream>

using namespace taco;

/*
B:  1  0  0  0
    0  0  2  3
    0  0  0  0

C:  0  4
    5  0
    0  0
    0  0

A:  0  4
    0  0
    0  0
*/

int main(int argc, char* argv[])
{
    // Create formats
    Format csr({ Dense, Sparse }, { 0, 1 });
    Format csc({ Dense, Sparse }, { 1, 0 });

    // Create tensors
    Tensor<double> A({ 3, 2 }, csr);
    Tensor<double> B({ 3, 4 }, csr);
    Tensor<double> C({ 4, 2 }, csr);

    // Insert data into B and C
    B.insert({ 0, 0 }, 1.0);
    B.insert({ 1, 0 }, 3.0);
    B.insert({ 1, 1 }, 3.0);
    B.insert({ 1, 3 }, 3.0);
    C.insert({ 0, 1 }, 10.0);
    C.insert({ 1, 0 }, 5.0);

    // Pack data as described by the formats
    B.pack();
    C.pack();
    // Form a matrix multiplication expression
    IndexVar i, j, k;
    A(i, j) = B(i, k) * C(k, j);

    // Compile the expression
    A.compile();

    // Assemble A's indices and numerically compute the result
    A.assemble();
    A.compute();

    std::cout << A << std::endl;
}