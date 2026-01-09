#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include "autograd.h"
#include "tensor.h"
#include "nditerator.h"

class CrossEntropyLossFn {
    
    double error;

    public:
        CrossEntropyLossFn(std::shared_ptr<TensorX> logits, std::shared_ptr<TensorX> target, string op){
            Tensor logit = logits->get_data().log_softmax(logits.ndim()-1);   
            Tensor trg = target->get_data();

            double loss = 0.0;
            size_t count = 0;
            auto indices = NDRange(trg.shape());
            for(const auto& idx: indices){
                std::vector<size_t> index = idx;
                index.push_back(static_cast<size_t>(trg.at(idx)));
                loss -= logit.at(index);
                count++;
            }
            error = loss / count; 
        }
};
#endif
