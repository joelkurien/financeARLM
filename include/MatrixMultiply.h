#ifndef MATRIXMULTIPLY_H
#define MATRIXMULTIPLY_H

#include<iostream>
#include<vector>
#include<thread>
#include<cblas.h>
#include "tensor.h"

class MatrixMul {
    private: 
        const CBLAS_LAYOUT layout = CblasRowMajor;
        const CBLAS_TRANSPOSE transA = CblasNoTrans;
        const CBLAS_TRANSPOSE transB = CblasNoTrans;

        const double alpha = 1;
        const double beta = 0;

        Tensor batch_multiplication(const Tensor& a, const Tensor& b, const std::vector<size_t> batch_shape);

        Tensor single_multiplication(const Tensor& a, const Tensor& b);

        Tensor broadcast_multiplication(const Tensor& a, const Tensor& b);

    public:
        static Tensor matmul(Tensor a, Tensor b);
};


#endif 
