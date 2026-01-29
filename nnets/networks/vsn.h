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

    size_t n_features;
    GRN gating_grn_layer;
    
    public:
        VSN(size_t units, double dr, size_t n_feat)
            : nodes(units), dropout_rate(dr), n_features(n_feat), 
              weight_projection(n_feat, true), gating_grn_layer(units, dr)
        {
            for(size_t feat = 0; feat < n_features; feat++){
                grn_layers.emplace_back(GRN(nodes, dropout_rate));
                linear_layers.emplace_back(Linear(nodes, true));
            }
        }

        std::shared_ptr<TensorX> numerical_encoding_feature(const size_t idx, std::shared_ptr<TensorX> input) {
            return linear_layers[idx].forward(input);
        }

        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            size_t ld = input->get_data().ndim()-1;
            size_t n_features = input->get_data().shape()[ld];

            std::vector<std::shared_ptr<TensorX>> features = chunk(input, n_features, ld);
            // for(size_t feat = 0; feat < n_features; feat++){
            //     features[feat] = numerical_encoding_feature(feat, features[feat]);
            // }
            
            std::shared_ptr<TensorX> concat_input = concat(features, ld);
            std::shared_ptr<TensorX> weights = gating_grn_layer.forward(concat_input);
            std::shared_ptr<TensorX> weights_proj = weight_projection.forward(weights); 
            weights_proj = Softmax(ld, true).forward(weights_proj);
            
            std::shared_ptr<TensorX> expanded_weights = unsqueeze(weights_proj, ld+1);

            std::vector<std::shared_ptr<TensorX>> grn_results;
            std::vector<std::shared_ptr<TensorX>> feature_vecs = chunk(concat_input, n_features, ld);
            for(size_t feat = 0; feat < n_features; feat++){
                grn_results.push_back(grn_layers[feat].forward(feature_vecs[feat]));
            }
            
            std::shared_ptr<TensorX> results = stack(grn_results, ld);
            prnt(weights_proj->get_data().shape());
            prnt(results->get_data().shape());
            return squeeze(matmul(transpose(weights_proj), results), ld);
        }

        virtual std::vector<std::shared_ptr<TensorX>> parameters() override {
            std::vector<std::shared_ptr<TensorX>> params;

            auto append_params = [&](RootLayer& layer){
                params.insert(params.end(), layer.parameters().begin(), layer.parameters().end());
            };
            
            append_params(weight_projection);
            append_params(gating_grn_layer);

           for(GRN grn_layer: grn_layers) append_params(grn_layer);
           for(Linear linear_layer: linear_layers) append_params(linear_layer);

           return params;
        }
};

#endif
