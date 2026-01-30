#ifndef STATIC_ENCODER_H
#define STATIC_ENCODER_H

#include "autograd.h"
#include "grn.h"
#include "vsn.h"

struct Static_Contexts {
    std::shared_ptr<TensorX> cs;
    std::shared_ptr<TensorX> cc;
    std::shared_ptr<TensorX> ch;
    std::shared_ptr<TensorX> cv;
};

class Static_Encoder : public RootLayer {
    GRN grn_layer_s;
    GRN grn_layer_c;
    GRN grn_layer_h;
    GRN grn_layer_e;
    VSN selection_layer;

    public:
        Static_Encoder (size_t units, double dr, size_t in_feat, size_t n_feat)
            : selection_layer(units, dr, in_feat, n_feat),
              grn_layer_s (units, dr, units), 
              grn_layer_c (units, dr, units), 
              grn_layer_h (units, dr, units), 
              grn_layer_e (units, dr, units) {}

        Static_Contexts forward_all(std::shared_ptr<TensorX> input){
            std::shared_ptr<TensorX> vsn_selection = selection_layer.forward(input);
            std::shared_ptr<TensorX> c_s = grn_layer_s.forward(vsn_selection);
            std::shared_ptr<TensorX> c_c = grn_layer_c.forward(vsn_selection);
            std::shared_ptr<TensorX> c_h = grn_layer_h.forward(vsn_selection);
            std::shared_ptr<TensorX> c_e = grn_layer_e.forward(vsn_selection);
            return {c_s, c_c, c_h, c_e};
        }
        
        virtual std::vector<std::shared_ptr<TensorX>> parameters(){
            std::vector<std::shared_ptr<TensorX>> params;
            auto append_params = [&](RootLayer& layer){
                if(!layer.parameters().empty()){
                    for(std::shared_ptr<TensorX>& p: layer.parameters()){
                        params.push_back(p);
                    }
                }           
            };

            append_params(selection_layer);
            append_params(grn_layer_s);
            append_params(grn_layer_c);
            append_params(grn_layer_h);
            append_params(grn_layer_e);

            return params;
        }
};

#endif
