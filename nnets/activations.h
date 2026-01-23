#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "autograd.h"
#include "RootLayer.h"

#define NO_ATTRIBUTES virtual std::vector<std::shared_ptr<TensorX>> parameters() override { return {}; }

class ReLU : public RootLayer {
    public:
        NO_ATTRIBUTES
        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            return relu(input);
        }
        ~ReLU() override = default;
};

class Sigmoid : public RootLayer {
    public:
        NO_ATTRIBUTES
        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            return sigmoid(input);
        }

        ~Sigmoid() override = default;
};

class GLU : public RootLayer {
    public:
        NO_ATTRIBUTES
        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            return glu(input);
        }

        ~GLU() override = default;
};

class ReGLU : public RootLayer {
    public:
        NO_ATTRIBUTES
        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            return reGlu(input);
        }

        ~ReGLU() override = default;

};

class Tanh : public RootLayer {
    public:
        NO_ATTRIBUTES
        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            return tanh(input);
        }

        ~Tanh() override = default;
};

class ELU : public RootLayer {
    double alpha;
    public:
        NO_ATTRIBUTES
        ELU(double a) : alpha(a) {}

        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            return elu(input, alpha);
        }

        ~ELU() override = default;
};
#endif
