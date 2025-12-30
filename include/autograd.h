#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include<vector>
#include<memory>
#include<iostream>
#include<functional>
#include<algorithm>
#include<unordered_set>
#include "tensor.h"
#include "MatrixMultiply.h"

class TensorX;

class Autograd {
    public:
        std::function<void()> backward_function;
        std::vector<std::shared_ptr<TensorX>> inputs;
        Autograd(std::function<void()> backward_fn, std::vector<std::shared_ptr<TensorX>> autograd_inputs);
};

class TensorX : public std::enable_shared_from_this<TensorX>{
    Tensor data_;
    Tensor grad_;
    bool required_grad;
    std::shared_ptr<Autograd> autograd_function;

    public:
        TensorX(Tensor data, bool req_grad);
        
        Tensor& get_data();
        //const Tensor& get_data() const { return data_; }
        Tensor& get_grad();
        //const Tensor& get_grad() const { return grad_; }
        bool get_required_grad() const;
        void set_autograd_fn(std::shared_ptr<Autograd> fn);
        void accumulate(Tensor& new_grad);
        void grad_zeros();
        void backward(const std::optional<Tensor>& grad = std::nullopt);

    private:
        void topological_sort(std::shared_ptr<TensorX> tensor, std::unordered_set<TensorX*>& visited, 
                                std::vector<std::shared_ptr<TensorX>>& topo_order);
};

//elementwise operations
std::shared_ptr<TensorX> multiply(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y);
std::shared_ptr<TensorX> multiply(std::shared_ptr<TensorX> x, double y);
std::shared_ptr<TensorX> add(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y);
std::shared_ptr<TensorX> add(std::shared_ptr<TensorX> x, double y);
std::shared_ptr<TensorX> subtract(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y);
std::shared_ptr<TensorX> subtract(std::shared_ptr<TensorX> x, double y);
std::shared_ptr<TensorX> divide(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y);
std::shared_ptr<TensorX> divide(std::shared_ptr<TensorX> x, double y);

std::shared_ptr<TensorX> sqrt(std::shared_ptr<TensorX> x);
std::shared_ptr<TensorX> exp(std::shared_ptr<TensorX> x);
std::shared_ptr<TensorX> log(std::shared_ptr<TensorX> x);
std::shared_ptr<TensorX> pow(std::shared_ptr<TensorX> x, const double n);

//functional operations
std::shared_ptr<TensorX> softmax(std::shared_ptr<TensorX> x, const size_t axis);
std::shared_ptr<TensorX> layer_norm(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> gamma, std::shared_ptr<TensorX> beta, const size_t axis);
std::shared_ptr<TensorX> relu(std::shared_ptr<TensorX> x);
std::shared_ptr<TensorX> gelu(std::shared_ptr<TensorX> x);
std::shared_ptr<TensorX> sum(std::shared_ptr<TensorX> x, const size_t axis);
std::shared_ptr<TensorX> mean(std::shared_ptr<TensorX> x, const size_t axis);
std::shared_ptr<TensorX> var(std::shared_ptr<TensorX> x, const size_t axis);

std::shared_ptr<TensorX> matmul(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y);
std::shared_ptr<TensorX> transpose(std::shared_ptr<TensorX> x);
std::shared_ptr<TensorX> permute(std::shared_ptr<TensorX> x, const std::optional<std::vector<size_t>>& rotaxis = std::nullopt);
std::shared_ptr<TensorX> reshape(std::shared_ptr<TensorX> x, std::vector<size_t> new_shape);
std::shared_ptr<TensorX> concat(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y, const size_t axis);
 
// cross_entropy_loss
// std::shared_ptr<TensorX> MSELoss(std::shared_ptr<TensorX> x);
#endif
