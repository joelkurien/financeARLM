#ifndef TENSOR_FAC
#define TENSOR_FAC

#include "autograd.h"
#include<memory>

namespace tensor {
    std::shared_ptr<TensorX> create(Tensor data, bool requires_grad = false){
        return std::make_shared<TensorX>(data, requires_grad);
    }

    std::shared_ptr<TensorX> deep_create(std::vector<size_t> shape, bool requires_grad = false){
        return std::make_shared<TensorX>(Tensor(shape), requires_grad);
    }
}

#endif