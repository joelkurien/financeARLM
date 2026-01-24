#ifndef GRN_H
#define GRN_H

#include "autograd.h"
#include "Linear.h"
#include "activations.h"
#include "RootLayer.h"
#include "normalizations.h"
#include "dropout.h"

class GRN {
    RootLayer* layer;
    public:
        GRN() = default;

        std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) {
            layer->set(Linear(512).forward(input));
            layer->set(ELU(1).forward(layer->data()));
            layer->set(Linear(256).forward(layer->data()));
            layer->set(Dropout(0.3).forward(layer->data()));
            layer->set(add(input, GLU().forward(layer->data())));
            layer->set(LayerNormalization().forward(layer->data()));
            return layer->data();
        }
};
#endif
