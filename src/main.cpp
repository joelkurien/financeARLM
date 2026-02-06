#include <vector>
#include "tensor.h"
#include "autograd.h"
#include "tensor_fac.h"
#include "grn.h"
#include "vsn.h"
#include "static_encoder.h"
#include "AdamWOptimizer.h"
#include "lstm.h"
#include "interoperable_mha.h"
#include "CosineAnnealing.h"

class JankyTFTLayer : public RootLayer {
    public:
        LSTM rnn;
        Interpretable_MHA attention;
        GRN grn;
        size_t hidden_dims;

        JankyTFTLayer(size_t in_feat, size_t out_feat, size_t num_heads)
            : hidden_dims(out_feat),
              rnn(in_feat, out_feat, in_feat, true),
              attention(out_feat, num_heads, true),
              grn(in_feat, out_feat, 0.3)
        {}

        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            size_t batch_size = input->get_data().shape().front();
            
            std::shared_ptr<TensorX> h0 = tensor::create(Tensor({batch_size, hidden_dims}), true);
            std::shared_ptr<TensorX> c0 = tensor::create(Tensor({batch_size, hidden_dims}), true);

            // size_t seq_len = input->get_data().shape()[1];
            // auto sequence = rnn.seq2seq_forward(input, h0, c0, seq_len);
            auto grn_res = grn.forward(input);
            return attention.forward(grn_res, 1);
        }

        virtual std::vector<std::shared_ptr<TensorX>> parameters() override {
            std::vector<std::shared_ptr<TensorX>> rnn_params = rnn.parameters();
            std::vector<std::shared_ptr<TensorX>> attn_params = attention.parameters();
            std::vector<std::shared_ptr<TensorX>> grn_params = grn.parameters();
            rnn_params.insert(rnn_params.end(), grn_params.begin(), grn_params.end());
            rnn_params.insert(rnn_params.end(), attn_params.begin(), attn_params.end());
            return rnn_params;
        }
};

void run_test() {
    // Hyperparameters
    size_t batch = 4;
    size_t seq = 10;
    size_t feat = 16;
    size_t hidden = 16;
    size_t total_steps = 50;

    // Initialize Components
    JankyTFTLayer model(feat, hidden, 4);
    CosineAnnealing scheduler(1e-3, total_steps, 10, 1e-5);
    AdamW optimizer(model.parameters(), 1e-3,0.9, 0.99, 1e-8, 0.01);

    std::cout << "Starting Training Test..." << std::endl;

    for (size_t i = 0; i < total_steps; i++) {
        // 1. Generate Fake Data {Batch, Seq, Feat}
        std::shared_ptr<TensorX>  input = tensor::create(Tensor({batch, seq, feat}), true);
        input->get_data().xavier_ud(batch, seq+feat);
        std::shared_ptr<TensorX> target = tensor::create(ones({batch, seq, hidden}), true);

        // 2. Forward & Loss
        std::shared_ptr<TensorX>  output = model.forward(input);
        std::shared_ptr<TensorX> loss = pinball_loss(output, target, 0.1); 
        prnt(loss->get_data().shape());
        prntd(loss->get_data().as_vector_const());
        // 3. Backprop
        optimizer.zero_grad();
        loss->backward();
        
        // 4. Update LR and Step
        // double current_lr = scheduler.get_learning_rate();
        // std::cout<<current_lr<<std::endl;
        optimizer.set_lr(0.01);
        
        optimizer.step();
        scheduler.step();

        std::cout << "Step: " << i << " | Loss: " << loss->get_data().at({0,0}) << std::endl;
    }
}

int main(){
    run_test(); 

    return 0;
}
