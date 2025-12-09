#ifndef MATRIXMULTIPLY_H
#define MATRIXMULTIPLY_H

#include<iostream>
#include<vector>
#include<thread>
#include<cblas.h>
#include "tensor.h"

using namespace std;

class MatrixMul {
    private: 
        const CBLAS_LAYOUT layout = CblasRowMajor;
        const CBLAS_TRANSPOSE transA = CblasNoTrans;
        const CBLAS_TRANSPOSE transB = CblasNoTrans;

        const double alpha = 1;
        const double beta = 0;

        Tensor batch_multiplication(const Tensor& a, const Tensor& b){
            vector<size_t> a_shape = a.shape();
            vector<size_t> b_shape = b.shape();

            if(a_shape[0] != b_shape[0]) throw invalid_argument("Batch size mismatch");

            size_t B = a_shape[0];
            int M = static_cast<int>(a_shape[1]);
            int K = static_cast<int>(a_shape[2]);
            int N = static_cast<int>(b_shape[2]);

            const int stride_a = M*K;
            const int stride_b = K*N;
            const int stride_c = M*N;

            Tensor c({B, a_shape[1], b_shape[2]});
            for(size_t i=0; i<B; i++){
                const double* a_ptr = a.data() + i* stride_a;
                const double* b_ptr = b.data() + i* stride_b;
                double* c_ptr = c.data() + i*stride_c;

                cblas_dgemm(layout, transA, transB, M, N, K, 
                            alpha, a_ptr, K, b_ptr, N, beta, c_ptr, N);
            }
            return c;
        }

        Tensor single_multiplication(const Tensor& a, const Tensor& b){
            vector<size_t> a_shape = a.shape();
            vector<size_t> b_shape = b.shape();

            int M = static_cast<int>(a_shape[0]);
            int K = static_cast<int>(a_shape[1]);
            int N = static_cast<int>(b_shape[1]);

            Tensor c({a_shape[0], b_shape[1]});
            cblas_dgemm(layout, transA, transB, M, N, K, 
                        alpha, a.data(), K, b.data(), N, beta, c.data(), N);
            return c;
        }

        Tensor broadcast_multiplication(const Tensor& a, const Tensor& b){
            vector<size_t> a_shape = a.shape();
            vector<size_t> b_shape = b.shape();
            
            size_t B = a_shape[0];
            int M = static_cast<int>(a_shape[1]);
            int K = static_cast<int>(a_shape[2]);
            int N = static_cast<int>(b_shape[1]);

            const int stride_a = M*K;
            const int stride_c = M*N;

            Tensor c({B, a_shape[1], b_shape[1]});
            for(size_t i=0; i<B; i++){
                const double* a_ptr = a.data() + i* stride_a;
                double* c_ptr = c.data() + i*stride_c;

                cblas_dgemm(layout, transA, transB, M, N, K, 
                            alpha, a_ptr, K, b.data(), N, beta, c_ptr, N);
            }
            return c;
        }

    public:
        static Tensor matmul(Tensor a, Tensor b){
            Tensor c;
            const vector<size_t> shape_a = a.shape();
            const vector<size_t> shape_b = b.shape();

            MatrixMul mul;
            if(a.ndim() >= 3 && b.ndim() >= 3){
                vector<size_t> a_shape = {0,shape_a[a.ndim()-2], shape_a[a.ndim()-1]};
                vector<size_t> b_shape= {0,shape_b[b.ndim()-2], shape_b[b.ndim()-1]};

                a_shape[0] = accumulate(shape_a.begin(), shape_a.end()-2, size_t{1}, multiplies<size_t>());
                a.reshape(a_shape);

                b_shape[0] = accumulate(shape_b.begin(), shape_b.end()-2, size_t{1}, multiplies<size_t>());
                b.reshape(b_shape);

                if(a_shape[0] != b_shape[0] || a_shape[2] != b_shape[1]) {
                    throw runtime_error("Incompatible shapes");
                }
                c = mul.batch_multiplication(a,b);
                a.reshape(shape_a);
                b.reshape(shape_b);
            }   
            else if(a.ndim() == 2 && b.ndim() == 2) {
                if(shape_a[1] != shape_b[0]) {
                    throw runtime_error("Incompatible shapes");
                }
                c = mul.single_multiplication(a,b);
            }
            else if(a.ndim() >=3 && b.ndim() == 2){
                vector<size_t> a_shape = {0,shape_a[a.ndim()-2], shape_a[a.ndim()-1]};
                a_shape[0] = accumulate(shape_a.begin(), shape_a.end()-2, size_t{1}, multiplies<size_t>());
                a.reshape(a_shape);
                if(a_shape[2] != shape_b[0]) {
                    throw runtime_error("Incompatible shapes");
                }
                c = mul.broadcast_multiplication(a,b);
                a.reshape(shape_a);
            } else {
                throw runtime_error("This dimensional multiplication is not currently available, maybe soon many be never, who knows");
            }
            return c;
        }
};

#endif 