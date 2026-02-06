#include "autograd.h"
#include "RootLayer.h"
#include "Linear.h"
#include "tensor.h"
#include "tensor_fac.h"

class Interpretable_MHA: public RootLayer {
    size_t d_model;
    size_t num_heads;
    bool initialize;
    size_t d_k;

    Linear QKV_layer;
    Linear output_proj;

    std::shared_ptr<TensorX> mask_cache;
    public:
        Interpretable_MHA(size_t units, size_t heads, bool init = false)
            : d_model(units), num_heads(heads), initialize(init),
              QKV_layer(d_model, 3*d_model, initialize),
              output_proj(d_model, d_model),
              d_k(d_model/num_heads) {}

        std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input, const size_t axis, 
                                         std::optional<double> mask = std::nullopt){
            std::vector<std::shared_ptr<TensorX>> QKV = chunk(QKV_layer.forward(input), 3, 2);
            std::shared_ptr<TensorX> Q = QKV[0];
            std::shared_ptr<TensorX> K = QKV[1];
            std::shared_ptr<TensorX> V = QKV[2];
            
            std::vector<size_t> shapes = input->get_data().shape();
            size_t batch = shapes[0];
            size_t seq_len = shapes[1];
            size_t feats = shapes[2];

            V = transpose(reshape(V, {batch, seq_len, num_heads, d_k}), 1, 2);

            std::shared_ptr<TensorX> score = divide(matmul(Q, transpose(K)),std::sqrt(d_k));

            if(!mask_cache || mask_cache->get_data().shape()[2] != seq_len){
                std::vector<double> mask_vec(seq_len*seq_len);
                for(size_t i=0; i<seq_len; i++){
                    for(size_t j=0; j<seq_len; j++){
                        mask_vec[i*seq_len + j] = j>=i ? 0.0 : 1.0;
                    }
                }

                Tensor bare_mask(mask_vec, {seq_len, seq_len});
                bare_mask = bare_mask.unsqueeze(0);
                mask_cache = tensor::create(bare_mask, true);
            }
            
            Tensor fill_mask = mask_cache->get_data().expand({batch, seq_len, seq_len});
            score = masked_fill(score, fill_mask, mask.value_or((double)-1e9));
            std::shared_ptr<TensorX> attention = softmax(score, score->get_data().ndim()-1);
            std::shared_ptr<TensorX> expanded_attn = unsqueeze(attention, 1);
            expanded_attn = expand(expanded_attn, {batch, num_heads, seq_len, seq_len});
            std::shared_ptr<TensorX> context = matmul(expanded_attn, V);

            context = reshape(transpose(context, 1, 2), {batch, seq_len, d_model});
            std::shared_ptr<TensorX> output = output_proj.forward(context);
            return output;
        }

        virtual std::vector<std::shared_ptr<TensorX>> parameters() {
            std::vector<std::shared_ptr<TensorX>> params;

            auto append_params = [&](RootLayer& layer){
                if(!layer.parameters().empty()){
                    for(std::shared_ptr<TensorX>& p: layer.parameters()){
                        params.push_back(p);
                    }
                }
            };

            append_params(QKV_layer);
            append_params(output_proj);

            return params;
        }
};
