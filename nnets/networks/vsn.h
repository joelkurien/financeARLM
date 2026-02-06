#ifndef VSH_H
#define VSH_H

#include "autograd.h"
#include "tensor.h"
#include "grn.h"
#include "RootLayer.h"
#include "activations.h"
#include "Linear.h"

/*
 * Variable Selection Network
 * */
class VSN: public RootLayer {
    size_t nodes;
    double dropout_rate;
    std::vector<GRN> grn_layers;
    std::vector<Linear> linear_layers;
    Linear weight_projection;

    size_t in_features;
    size_t n_features;
    GRN gating_grn_layer;
    
    public:
        VSN(size_t units, double dr, size_t in_feat, size_t n_feat)
            : nodes(units), dropout_rate(dr), n_features(n_feat), 
              in_features(in_feat),
              gating_grn_layer(units, units*n_feat, dr), 
              weight_projection(units, n_feat, true) 
        {
            for(size_t feat = 0; feat < n_features; feat++){
                grn_layers.emplace_back(GRN(nodes, nodes, dropout_rate));
                linear_layers.emplace_back(Linear(1, nodes, true));
            }
        }

        std::shared_ptr<TensorX> numerical_encoding_feature(const size_t idx, std::shared_ptr<TensorX> input) {
            return linear_layers[idx].forward(input);
        }

        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            size_t ld = input->get_data().ndim()-1;
            size_t n_features = input->get_data().shape()[ld];

            std::vector<std::shared_ptr<TensorX>> features = chunk(input, n_features, ld);
            for(size_t feat = 0; feat < n_features; feat++){
                features[feat] = numerical_encoding_feature(feat, features[feat]);
            }
            
            std::shared_ptr<TensorX> concat_input = concat(features, ld);
            std::shared_ptr<TensorX> weights = gating_grn_layer.forward(concat_input);
            std::shared_ptr<TensorX> weights_proj = weight_projection.forward(weights); 
            weights_proj = Softmax(ld, true).forward(weights_proj);
           
            std::shared_ptr<TensorX> expanded_weights = unsqueeze(weights_proj, ld); 

            std::vector<std::shared_ptr<TensorX>> grn_results;
            std::vector<std::shared_ptr<TensorX>> feature_vecs = chunk(concat_input, n_features, ld);
            for(size_t feat = 0; feat < n_features; feat++){
                std::shared_ptr<TensorX> grn_res = grn_layers[feat].forward(feature_vecs[feat]);
                grn_results.push_back(grn_res);
            }
            
            std::shared_ptr<TensorX> results = stack(grn_results, ld); 
            std::shared_ptr<TensorX> mult_val = matmul(expanded_weights, results);
            return squeeze(mult_val, ld);
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
            
            append_params(weight_projection);
            append_params(gating_grn_layer);

           for(GRN& grn_layer: grn_layers) append_params(grn_layer);
           for(Linear& linear_layer: linear_layers) append_params(linear_layer);
            

           return params;
        }
};

#endif
