#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include "RootLayer.h"
#include "autograd.h"
#include "tensor_fac.h"
#include "tensor.h"

class LayerNormalization : public RootLayer {
    std::shared_ptr<TensorX> gamma;
    std::shared_ptr<TensorX> beta;
    
    size_t axis;
    size_t in_features;

    public:
        LayerNormalization(size_t in_feat) 
            : in_features(in_feat), gamma(nullptr), beta(nullptr) {
                gamma = tensor::create(ones({1, in_features}), true);
                beta = tensor::create(ones({1, in_features}), true);
            }

        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override{
            try{
                axis = input->get_data().ndim()-1;
                return layer_norm(input, gamma, beta, axis);
            }
            catch(const std::runtime_error& rerr){
                std::cout<<"Runtime Error in Layer Normalization Layer: "<<rerr.what()<<std::endl;
            }
            catch(const std::invalid_argument& iarg){
                std::cout<<"Invalid Argument in Layer Normalization Layer: "<<iarg.what()<<std::endl;
            }
            catch(const std::exception& err){
                std::cout<<"Error in Layer Normalization Layer: "<<err.what()<<std::endl;
            }
            
            return input;
        }

        virtual std::vector<std::shared_ptr<TensorX>> parameters() override {
            return {gamma, beta};
        }
};

#endif
