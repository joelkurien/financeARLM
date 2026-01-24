#ifndef DROPOUT_H
#define DROPOUT_H

#include "RootLayer.h"
#include "autograd.h"

#define NO_ATTRIBUTES virtual std::vector<std::shared_ptr<TensorX>> parameters() override { return {}; }

class Dropout : RootLayer {
    double prob;
    public:
        NO_ATTRIBUTES
        Dropout(double p)
            : RootLayer(), prob(p) {}
        
        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override{
            Tensor mask = Tensor(input->get_data().shape());
            return dropout(input, prob, true, mask);
        }
};

#endif
