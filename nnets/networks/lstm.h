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
    bool initialize;

    std::vector<Linear> linear_layers;
    
    public:
        LSTM(size_t in_feat, size_t out_feat, bool init)
            : in_features(in_feat), out_features(out_feat), initialize(init)
        {
            for(int i=0; i<4; i++){
                linear_layers.push_back(Linear(in_features, out_features, initialize));
                linear_layers.push_back(Linear(out_features, out_features, initialize));
            }

        }
        
        std::vector<LSTM_Contexts> forward(std::shared_ptr<TensorX> input, 
                                            std::shared_ptr<TensorX> h_0, 
                                            std::shared_ptr<TensorX> cell_0,
                                            size_t num_splits){
            
            std::shared_ptr<TensorX> forget_x = linear_layers[0].forward(input);
            std::shared_ptr<TensorX> ignore_x = linear_layers[2].forward(input);
            std::shared_ptr<TensorX> candidate_x = linear_layers[4].forward(input);
            std::shared_ptr<TensorX> filter_x = linear_layers[6].forward(input);

            std::vector<std::shared_ptr<TensorX>> forget_x_list = chunk(forget_x, num_splits, 0);
            std::vector<std::shared_ptr<TensorX>> ignore_x_list = chunk(ignore_x, num_splits, 0);
            std::vector<std::shared_ptr<TensorX>> candidate_x_list = chunk(candidate_x, num_splits, 0);
            std::vector<std::shared_ptr<TensorX>> filter_x_list = chunk(filter_x, num_splits, 0);
            
            std::vector<LSTM_Contexts> lstm_contexts_list = {{h_0, cell_0}};
            std::shared_ptr<TensorX> h_prev = h_0;
            std::shared_ptr<TensorX> cell_prev = cell_0;
            
            for(size_t idx=0; idx<num_splits; idx++){
                std::shared_ptr<TensorX> f = Sigmoid().forward(add(forget_x_list[idx], linear_layers[1].forward(h_prev)));
                std::shared_ptr<TensorX> i = Sigmoid().forward(add(ignore_x_list[idx], linear_layers[3].forward(h_prev)));
                std::shared_ptr<TensorX> c = Tanh().forward(add(candidate_x_list[idx], linear_layers[5].forward(h_prev)));
                std::shared_ptr<TensorX> o = Sigmoid().forward(add(filter_x_list[idx], linear_layers[7].forward(h_prev)));

                cell_prev = add(multiply(f, cell_prev), multiply(i, c));
                h_prev = multiply(o, tanh(cell_prev));

                lstm_contexts_list.push_back({h_prev, cell_prev});
            }

            return lstm_contexts_list; 
        }

        virtual std::vector<std::shared_ptr<TensorX>> parameters() override {
            std::vector<std::shared_ptr<TensorX>> params;
            
            auto append_params = [&](RootLayer& layer){
                for(std::shared_ptr<TensorX>& p: layer.parameters()){
                    params.push_back(p);
                }
            };
            
            for(int i=0; i<8; i++){
                append_params(linear_layers[i]);
            } 
            return params;
        }
};
