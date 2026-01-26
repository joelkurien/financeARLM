#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "autograd.h"
#include "RootLayer.h"

#define NO_ATTRIBUTES virtual std::vector<std::shared_ptr<TensorX>> parameters() override { return {}; }

class ReLU : public RootLayer {
    public:
        ReLU() : RootLayer() {}
        NO_ATTRIBUTES
        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            try{
                return relu(input);
            }
            catch(const std::exception& err){
                std::cout<<"Error in ReLU Layer"<<err.what()<<std::endl;
            }
            return input;
        }
        ~ReLU() override = default;
};

class GeLU : public RootLayer {
    public:
        GeLU() : RootLayer() {}
        NO_ATTRIBUTES
        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            try{
                return gelu(input);
            }
            catch(const std::exception& err){
                std::cout<<"Error in GeLU Layer"<<err.what()<<std::endl;
            }
            return input;
        }
        ~GeLU() override = default;
};

class Sigmoid : public RootLayer {
    public:
        Sigmoid() : RootLayer() {}
        NO_ATTRIBUTES
        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            try{
                return sigmoid(input);
            }
            catch(const std::exception& err){
                std::cout<<"Error in Sigmoid Layer"<<err.what()<<std::endl;
            }
            return input;
        }

        ~Sigmoid() override = default;
};

class GLU : public RootLayer {
    public:
        GLU() : RootLayer() {}
        NO_ATTRIBUTES
        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            try{
                size_t last_dim = input->get_data().ndim()-1;
                return glu(input, last_dim);
            }
            catch(const std::exception& err){
                std::cout<<"Error in GLU Layer"<<err.what()<<std::endl;
            }
            return input;
        }

        ~GLU() override = default;
};

class ReGLU : public RootLayer {
    public:
        ReGLU() : RootLayer() {}
        NO_ATTRIBUTES
        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            try{
                return reGlu(input);
            }
            catch(const std::exception& err){
                std::cout<<"Error in ReGLU Layer"<<err.what()<<std::endl;
            }
            return input;
        }

        ~ReGLU() override = default;

};

class Tanh : public RootLayer {
    public:
        Tanh() : RootLayer() {}
        NO_ATTRIBUTES
        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            try{
                return tanh(input);
            }
            catch(const std::exception& err){
                std::cout<<"Error in tanh Layer"<<err.what()<<std::endl;
            }
            return input;
        }

        ~Tanh() override = default;
};

class ELU : public RootLayer {
    double alpha;
    public:
        ELU() : RootLayer() {}
        NO_ATTRIBUTES
        ELU(double a) : alpha(a) {}

        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            try{
                return elu(input, alpha);
            }
            catch(const std::exception& err){
                std::cout<<"Error in elu Layer"<<err.what()<<std::endl;
            }
            return input;
        }

        ~ELU() override = default;
};

class Softmax : public RootLayer {
    size_t axis;
    bool is_stable;
    public:
        Softmax() : RootLayer() {}
        NO_ATTRIBUTES
        Softmax(size_t a, bool stable) 
            : axis(a), is_stable(stable) {}

        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            try{
                return is_stable ? log_softmax(input, axis) : softmax(input, axis);
            }
            catch(const std::exception& err){
                std::cout<<"Error in tanh Layer"<<err.what()<<std::endl;
            }
            return input;
        }

        ~Softmax() override = default;
};
#endif
