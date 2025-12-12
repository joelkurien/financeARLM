#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include<vector>
#include<memory>
#include<iostream>
#include<functional>
#include<algorithm>
#include<unordered_set>
#include "tensor.h"

class TensorX;

class Autograd {
    public:
        std::function<void()> backward_function;
        std::vector<std::shared_ptr<TensorX>> inputs;
        Autograd(std::function<void()> backward_fn, std::vector<std::shared_ptr<TensorX>> autograd_inputs)
            : backward_function(backward_fn), inputs(autograd_inputs) {}
};

class TensorX : public std::enable_shared_from_this<TensorX>{
    Tensor data_;
    Tensor grad_;
    bool required_grad;
    std::shared_ptr<Autograd> autograd_function;

    public:
        TensorX(Tensor data, bool req_grad)
            : data_(data), required_grad(req_grad) 
        {
            if(!required_grad){
                grad_ = Tensor(data_.shape());
            }
        }
        
        Tensor& get_data()  { return data_; }
        const Tensor& get_data() const { return data_; }
        Tensor& get_grad() { return grad_; }
        const Tensor& get_grad() const { return grad_; }
        bool get_required_grad() const { return required_grad; }

        void set_autograd_fn(std::shared_ptr<Autograd> fn){
            autograd_function = fn;
        }

        void accumulate(Tensor& new_grad){
            if(!required_grad) return;
            grad_ = grad_ + new_grad;
        }        

        void grad_zeros() {
            if(required_grad)
                grad_ = Tensor(data_.shape());
        }

        void backward() {
            if(!required_grad) throw std::runtime_error("Gradient requirement was set to false");

            //initialize our gradient
            std::vector<double> ones (data_.size(), 1);
            grad_ = Tensor(ones, data_.shape());

            std::vector<std::shared_ptr<TensorX>> topological_order;
            std::unordered_set<TensorX*> visited;

            topological_sort(shared_from_this(), visited, topological_order);

            std::reverse(topological_order.begin(), topological_order.end());

            for(auto& tensor: topological_order){
                if((tensor.get())->autograd_function){
                    (tensor.get())->autograd_function->backward_function();
                }
            }
        }

    private:
        void topological_sort(std::shared_ptr<TensorX> tensor, std::unordered_set<TensorX*>& visited, std::vector<std::shared_ptr<TensorX>>& topo_order) {
            if(visited.find(tensor.get()) != visited.end()) return;

            visited.insert(tensor.get());

            if(tensor->autograd_function){
                for(auto& input: tensor->autograd_function->inputs){
                    topological_sort(input, visited, topo_order);
                }
            }

            topo_order.push_back(tensor);
        }
};

#endif