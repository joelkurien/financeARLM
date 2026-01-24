#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include "RootLayer.h"
#include "autograd.h"
#include "tensor_fac.h"
#include "tensor.h"

class LayerNormalization : RootLayer {
    std::shared_ptr<TensorX> gamma;
    std::shared_ptr<TensorX> beta;
    size_t axis;

    public:
        LayerNormalization() 
            : RootLayer(), gamma(nullptr), beta(nullptr) {}

        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override{
            if(gamma == nullptr && beta == nullptr){
                size_t last_dim = input->get_data().ndim()-1;
                size_t in_feat = input->get_data().shape()[last_dim];
                gamma = tensor::create(ones({1, in_feat}), true);
                beta = tensor::create(ones({1, in_feat}), true);
            }
            return layer_norm(input, gamma, beta, axis);
        }

        virtual std::vector<std::shared_ptr<TensorX>> parameters() override {
            return {gamma, beta};
        }
};

#endif
