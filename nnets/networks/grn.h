#ifndef GRN_H
#define GRN_H

#include "autograd.h"
#include "Linear.h"
#include "activations.h"
#include "normalizations.h"
#include "dropout.h"
#include "RootLayer.h"

/*
 * Gated Residual Network
 * */
class GRN : RootLayer {
    public:
        GRN(): RootLayer() {};

        Linear linear_layer_1{512, true};
        ELU elu_layer{1};
        Linear linear_layer_2{256, true};
        Dropout dropout_layer{0.3};
        GLU glu_layer;
        LayerNormalization layer_norm;

        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            std::shared_ptr<TensorX> x = linear_layer_1.forward(input);
            x = elu_layer.forward(x);
            x = linear_layer_2.forward(x);
            x = dropout_layer.forward(x);

            std::shared_ptr<TensorX> residual = add(input, glu_layer.forward(x));
            return layer_norm.forward(residual);
        }

        virtual std::vector<std::shared_ptr<TensorX>> parameters() override {
            std::vector<std::shared_ptr<TensorX>> params;
            std::vector<std::shared_ptr<TensorX>> l1_params = linear_layer_1.parameters();
            std::vector<std::shared_ptr<TensorX>> l2_params = linear_layer_2.parameters();
            std::vector<std::shared_ptr<TensorX>> norm_params = layer_norm.parameters();
            
            params.insert(params.end(), l1_params.begin(), l1_params.end());
            params.insert(params.end(), l2_params.begin(), l2_params.end());
            params.insert(params.end(), norm_params.begin(), norm_params.end());

            return params;
        }
};
#endif
