#ifndef LINEAR_H
#define LINEAR_H

#include "autograd.h"
#include "tensor_fac.h"
#include "RootLayer.h"

class Linear : public RootLayer {
    std::shared_ptr<TensorX> weights;
    std::shared_ptr<TensorX> bias;
    public:
        Linear(size_t in_features, size_t out_features, bool initialize = false){
            weights = tensor::deep_create({in_features, out_features}, true);
            bias = tensor::deep_create({1, out_features}, true);
            
            if(initialize){
                weights->get_data().xavier_ud(in_features, out_features);
            }
        }    

        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            std::shared_ptr<TensorX> linear_calculation = add(matmul(input, weights), bias);
            return linear_calculation;
        }

        virtual std::vector<std::shared_ptr<TensorX>> parameters() override {
            return {weights, bias};
        }

        ~Linear() override = default;

};

#endif
