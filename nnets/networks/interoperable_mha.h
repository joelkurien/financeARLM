#include "autograd.h"
#include "RootLayer.h"
#include "tensor.h"
#include "tensor_fac.h"

class Interpretable_MHA: public RootLayer {
    size_t d_model;
    size_t num_heads;
    bool initialize;
    size_t d_k;

    std::shared_ptr<TensorX> weights_Q;
    std::shared_ptr<TensorX> weights_K;
    std::shared_ptr<TensorX> weight_V;

    std::shared_ptr<TensorX> mask_cache;
    public:
        Interpretable_MHA(size_t units, size_t heads, bool init = false)
            : d_model(units), num_heads(heads), initialize(init)
        {
            weights_Q = tensor::create(Tensor({d_model, d_model}), true);
            weights_K = tensor::create(Tensor({d_model, d_model}), true);
            
            d_k = d_model/num_heads;
            weight_V = tensor::create(Tensor({d_model, d_k}), true);
            
            if(initialize){
                weights_Q->get_data().xavier_ud(d_model, d_model);
                weights_K->get_data().xavier_ud(d_model, d_model);
                weight_V->get_data().xavier_ud(d_model, d_k);
            }
        }

        std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input, const size_t axis, 
                                         std::optional<double> mask = std::nullopt){
            std::vector<std::shared_ptr<TensorX>> Qs = chunk(matmul(input, weights_Q), num_heads, axis);
            std::vector<std::shared_ptr<TensorX>> Ks = chunk(matmul(input, weights_K), num_heads, axis);
            std::shared_ptr<TensorX> V = matmul(input, weight_V);

            size_t idx = input->get_data().ndim();
            std::shared_ptr<TensorX> Q = stack(Qs, 1);
            std::shared_ptr<TensorX> K = stack(Ks, 1);
            
            std::shared_ptr<TensorX> argument = divide(matmul(Q, transpose(K)), sqrt(d_k));
            size_t arg_len = argument->get_data().ndim();
            size_t seq_len = argument->get_data().shape()[arg_len-2];
            if(!mask_cache || mask_cache->get_data().ndim() != arg_len 
               || mask_cache->get_data().shape()[arg_len-2] != seq_len){
                std::vector<double> mask_vec(seq_len*seq_len);
                for(size_t i=0; i<seq_len; i++){
                    for(size_t j=0; j<seq_len; j++){
                        mask_vec[i*seq_len + j] = j>=i ? 0.0 : 1.0;
                    }
                }

                Tensor bare_mask(mask_vec, {seq_len, seq_len});
                bare_mask = bare_mask.unsqueeze(0).unsqueeze(0);
                mask_cache = tensor::create(bare_mask, true);
            }
            
            Tensor fill_mask = mask_cache->get_data()
                .expand({argument->get_data().shape()[0], num_heads, seq_len, seq_len});
            argument = masked_fill(argument, fill_mask, mask.value_or((double)-1e9));

            std::shared_ptr<TensorX> attention = mean(softmax(argument, arg_len-1), 1);
            return matmul(attention, V);
        }

        virtual std::vector<std::shared_ptr<TensorX>> parameters() {
            return {weights_Q, weights_K, weight_V};
        }
};
