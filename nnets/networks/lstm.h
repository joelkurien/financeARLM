#include "autograd.h"
#include "RootLayer.h"
#include "Linear.h"
#include "activations.h"

struct LSTM_Contexts {
    std::shared_ptr<TensorX> hidden_t;
    std::shared_ptr<TensorX> cell_t;
};

class LSTM : public RootLayer {
    size_t in_features;
    size_t out_features;
    size_t target_features;
    bool initialize;

    std::vector<LSTM_Contexts> lstm_contexts_list;
    Linear w_gate;
    Linear u_gate;
    Linear seq2seq;
    public:
        LSTM(size_t in_feat, size_t out_feat, size_t trg_features, bool init=false)
            : in_features(in_feat), 
              out_features(out_feat), 
              initialize(init),
              target_features(trg_features),
              w_gate(in_feat, out_feat),
              u_gate(out_feat, 4*out_feat),
              seq2seq(out_features, target_features, initialize)
        {}
        
        void forward(std::shared_ptr<TensorX> input, 
                                            std::shared_ptr<TensorX> h_0, 
                                            std::shared_ptr<TensorX> cell_0,
                                            size_t num_splits){
            lstm_contexts_list.clear();
            lstm_contexts_list.reserve(num_splits);  

            std::shared_ptr<TensorX> data_x = w_gate.forward(input);
            std::shared_ptr<TensorX> h_prev = unsqueeze(h_0, input->get_data().ndim()-2);
            std::shared_ptr<TensorX> cell_prev = unsqueeze(cell_0, input->get_data().ndim()-2);
            std::vector<std::shared_ptr<TensorX>> x_chunks = chunk(data_x, num_splits, input->get_data().ndim()-2); 

            for(size_t idx=0; idx<num_splits; idx++){
                std::shared_ptr<TensorX> data_h = u_gate.forward(h_prev);

                std::shared_ptr<TensorX> add_gates = add(x_chunks[idx], data_h);
                std::vector<std::shared_ptr<TensorX>> gates = chunk(add_gates, 4, input->get_data().ndim()-1);
                
                auto f = sigmoid(gates[0]);
                auto i = sigmoid(gates[1]);
                auto c = tanh(gates[2]);
                auto o = sigmoid(gates[3]);

                cell_prev = add(multiply(f, cell_prev), multiply(i, c));
                h_prev = multiply(o, tanh(cell_prev));
               
                LSTM_Contexts ctx;
                ctx.hidden_t = h_prev;
                ctx.cell_t = cell_prev;
                lstm_contexts_list.push_back(ctx);
            }
        }

        std::shared_ptr<TensorX> seq2seq_forward(std::shared_ptr<TensorX> input, 
                                            std::shared_ptr<TensorX> h_0, 
                                            std::shared_ptr<TensorX> cell_0,
                                            size_t num_splits){
            forward(input, h_0, cell_0, num_splits);
            if(lstm_contexts_list.empty()) 
                throw std::runtime_error("There are no hidden states present to present an output");
            std::vector<std::shared_ptr<TensorX>> hidden_states;
            for(LSTM_Contexts& ctx: lstm_contexts_list){
                hidden_states.push_back(ctx.hidden_t);  
            }
            
            std::shared_ptr<TensorX> hidden_state = concat(hidden_states, input->get_data().ndim()-2);
            return seq2seq.forward(hidden_state);
        }

        virtual std::vector<std::shared_ptr<TensorX>> parameters() override {
            std::vector<std::shared_ptr<TensorX>> params;
            
            auto append_params = [&](RootLayer& layer){
                for(std::shared_ptr<TensorX>& p: layer.parameters()){
                    params.push_back(p);
                }
            };
            
            append_params(w_gate);
            append_params(u_gate);

            append_params(seq2seq);
            return params;
        }
};
