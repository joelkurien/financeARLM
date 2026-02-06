#ifndef TENSOR_FAC
#define TENSOR_FAC

#include "autograd.h"
#include<memory>

namespace tensor {
    inline std::shared_ptr<TensorX> create(Tensor data, bool requires_grad = true){
        return std::make_shared<TensorX>(data, requires_grad);
    }

    inline std::shared_ptr<TensorX> deep_create(std::vector<size_t> shape, bool requires_grad = true){
        return std::make_shared<TensorX>(Tensor(shape), requires_grad);
    }

    inline std::shared_ptr<TensorX> deep_create(std::vector<double> valvec, std::vector<size_t> shape, bool requires_grad = true){
        return std::make_shared<TensorX>(Tensor(valvec, shape), requires_grad);
    }
}

#endif
