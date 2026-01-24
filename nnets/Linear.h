#ifndef LINEAR_H
#define LINEAR_H

#include "autograd.h"
#include "tensor_fac.h"
#include "RootLayer.h"

class Linear : public RootLayer {
    std::shared_ptr<TensorX> weights;
    std::shared_ptr<TensorX> bias;
    bool initialize;
    size_t out_feat;
    public:
        Linear(size_t out_features, bool init = false) 
            : out_feat(out_features), initialize(init) {
            bias = tensor::deep_create({1, out_features}, true);
            
            
        }    

        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            size_t last_dim = input->get_data().ndim()-1;
            size_t in_features = input->get_data().shape()[last_dim];
            if(weights == nullptr){
                weights = tensor::deep_create({in_features, out_feat});
                if(initialize){
                    weights->get_data().xavier_ud(in_features, out_feat);
                }
            }
            std::shared_ptr<TensorX> linear_calculation = add(matmul(input, weights), bias);
            return linear_calculation;
        }

        virtual std::vector<std::shared_ptr<TensorX>> parameters() override {
            return {weights, bias};
        }

        ~Linear() override = default;

};

#endif
