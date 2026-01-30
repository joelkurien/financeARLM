#ifndef GRN_H
#define GRN_H

#include "autograd.h"
#include "tensor.h"
#include "Linear.h"
#include "activations.h"
#include "normalizations.h"
#include "dropout.h"
#include "RootLayer.h"

/*
 * Gated Residual Network
 * */
class GRN : public RootLayer {
    size_t nodes;
    double dropout_rate;
    Linear linear_layer_1;
    ELU elu_layer{1};
    Linear linear_layer_2;
    Dropout dropout_layer;
    Linear project_layer;
    GLU glu_layer;
    LayerNormalization layer_norm;

    size_t in_features;
    public:
        GRN( size_t units, double dr, size_t in_feat)
            : in_features(in_feat), nodes(units), dropout_rate(dr),
              linear_layer_1(in_feat, nodes, true), 
              linear_layer_2(nodes, nodes, true),
              dropout_layer(dropout_rate),
              project_layer(in_feat, nodes, true),
              layer_norm(nodes)
        {};

        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            std::shared_ptr<TensorX> x = linear_layer_1.forward(input);
            x = elu_layer.forward(x);
            size_t last_dim = input->get_data().ndim()-1;
            size_t out_features = input->get_data().shape()[last_dim];
            x = linear_layer_2.forward(x);
            x = dropout_layer.forward(x);
            if(input->get_data().shape().back() != nodes){
                input = project_layer.forward(input);
            } 
            std::shared_ptr<TensorX> residual = add(input, glu_layer.forward(x));
            residual = layer_norm.forward(residual);
            return residual;
        }

        virtual std::vector<std::shared_ptr<TensorX>> parameters() override {
            std::vector<std::shared_ptr<TensorX>> params;

            auto append_params = [&](RootLayer& layer){
                if(!layer.parameters().empty()){
                    for(std::shared_ptr<TensorX>& p: layer.parameters()){
                        params.push_back(p);
                    }
                }
            };           

            append_params(linear_layer_1);
            append_params(linear_layer_2);
            append_params(project_layer);
            append_params(layer_norm);

            std::cout<<params.size()<<std::endl;
            return params;
        }
};
#endif
