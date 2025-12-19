#include "MatrixMultiply.h"

Tensor MatrixMul::batch_multiplication(const Tensor& a, const Tensor& b){
    std::vector<size_t> a_shape = a.shape();
    std::vector<size_t> b_shape = b.shape();

    if(a_shape[0] != b_shape[0]) throw std::invalid_argument("Batch size mismatch");

    size_t B = a_shape[0];
    int M = static_cast<int>(a_shape[1]);
    int K = static_cast<int>(a_shape[2]);
    int N = static_cast<int>(b_shape[2]);

    const int stride_a = a.get_strides()[0];
    const int stride_b = b.get_strides()[0];
    const int stride_c = M*N;

    Tensor c({B, a_shape[1], b_shape[2]});
    for(size_t i=0; i<B; i++){
        const double* a_ptr = a.data() + i* stride_a;
        const double* b_ptr = b.data() + i* stride_b;
        double* c_ptr = c.data() + i*stride_c;

        int lda = static_cast<int>(a.get_strides()[1]);
        int ldb = static_cast<int>(b.get_strides()[1]);

        cblas_dgemm(layout, transA, transB, M, N, K, 
                    alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, N);
    }
    return c;
}

Tensor MatrixMul::single_multiplication(const Tensor& a, const Tensor& b){
    std::vector<size_t> a_shape = a.shape();
    std::vector<size_t> b_shape = b.shape();

    std::vector<size_t> a_strides = a.get_strides();
    std::vector<size_t> b_strides = b.get_strides();
    
    CBLAS_TRANSPOSE finalTransA = (a_strides.back() != 1) ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE finalTransB = (b_strides.back() != 1) ? CblasTrans : CblasNoTrans;

    int M = static_cast<int>(a_shape[0]);
    int K = static_cast<int>(a_shape[1]);
    int N = static_cast<int>(b_shape[1]);

    int lda = static_cast<int>(std::max(a_strides[0], a_strides[1]));
    int ldb = static_cast<int>(std::max(b_strides[0], b_strides[1]));

    Tensor c({a_shape[0], b_shape[1]});
    cblas_dgemm(layout, finalTransA, finalTransB, M, N, K, 
                alpha, a.data(), lda, b.data(), ldb, beta, c.data(), N);
    return c;
}

Tensor MatrixMul::broadcast_multiplication(const Tensor& a, const Tensor& b){
    std::vector<size_t> a_shape = a.shape();
    std::vector<size_t> b_shape = b.shape();
    
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

Tensor MatrixMul::matmul(Tensor a, Tensor b){
    Tensor c;
    const std::vector<size_t> shape_a = a.shape();
    const std::vector<size_t> shape_b = b.shape();

    MatrixMul mul;
    if(a.ndim() >= 3 && b.ndim() >= 3){
        std::vector<size_t> a_shape = {0,shape_a[a.ndim()-2], shape_a[a.ndim()-1]};
        std::vector<size_t> b_shape= {0,shape_b[b.ndim()-2], shape_b[b.ndim()-1]};

        a_shape[0] = std::accumulate(shape_a.begin(), shape_a.end()-2, size_t{1}, std::multiplies<size_t>());
        a.reshape(a_shape);

        b_shape[0] = std::accumulate(shape_b.begin(), shape_b.end()-2, size_t{1}, std::multiplies<size_t>());
        b.reshape(b_shape);

        if(a_shape[0] != b_shape[0] || a_shape[2] != b_shape[1]) {
            throw std::runtime_error("Incompatible shapes");
        }
        c = mul.batch_multiplication(a,b);
        a.reshape(shape_a);
        b.reshape(shape_b);
    }   
    else if(a.ndim() == 2 && b.ndim() == 2) {
        if(shape_a[1] != shape_b[0]) {
            throw std::runtime_error("Incompatible shapes");
        }
        c = mul.single_multiplication(a,b);
    }
    else if(a.ndim() >=3 && b.ndim() == 2){
        std::vector<size_t> a_shape = {0,shape_a[a.ndim()-2], shape_a[a.ndim()-1]};
        a_shape[0] = std::accumulate(shape_a.begin(), shape_a.end()-2, size_t{1}, std::multiplies<size_t>());
        a.reshape(a_shape);
        if(a_shape[2] != shape_b[0]) {
            throw std::runtime_error("Incompatible shapes");
        }
        c = mul.broadcast_multiplication(a,b);
        a.reshape(shape_a);
    } else {
        throw std::runtime_error("This dimensional multiplication is not currently available, maybe soon many be never, who knows");
    }
    return c;
}
