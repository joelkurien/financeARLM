#include "autograd.h"

Autograd::Autograd(std::function<void()> backward_fn, std::vector<std::shared_ptr<TensorX>> autograd_inputs)
    : backward_function(backward_fn), inputs(autograd_inputs) {}

TensorX::TensorX(Tensor data, bool req_grad)
            : data_(data), required_grad(req_grad) 
{
    if(required_grad){
        grad_ = Tensor(data_.shape());
    }
}
        
Tensor& TensorX::get_data()  { return data_; }
//const Tensor& TensorX::get_data() const { return data_; }
Tensor& TensorX::get_grad() { return grad_; }
//const Tensor& TensorX::get_grad() const { return grad_; }
bool TensorX::get_required_grad() const { return required_grad; }

void TensorX::set_autograd_fn(std::shared_ptr<Autograd> fn){
    autograd_function = fn;
}

void TensorX::accumulate(Tensor& new_grad){
    if(!required_grad) return;
    grad_ = grad_ + new_grad;
}        

void TensorX::grad_zeros() {
    if(required_grad)
        grad_ = Tensor(data_.shape());
}

void TensorX::backward() {
    if(!required_grad) throw std::runtime_error("Gradient requirement was set to false");

    //initialize our gradient
    std::vector<double> ones (data_.size(), 1);
    grad_ = Tensor(ones, data_.shape());

    std::vector<std::shared_ptr<TensorX>> topological_order;
    std::unordered_set<TensorX*> visited;

    topological_sort(shared_from_this(), visited, topological_order);

    std::reverse(topological_order.begin(), topological_order.end());

    for(auto& tensor: topological_order){
        if(tensor->autograd_function){
            tensor->autograd_function->backward_function();
        }
    }
}

void TensorX::topological_sort(std::shared_ptr<TensorX> tensor, std::unordered_set<TensorX*>& visited, 
                        std::vector<std::shared_ptr<TensorX>>& topo_order) {
    if(visited.find(tensor.get()) != visited.end()) return;

    visited.insert(tensor.get());

    if(tensor->autograd_function){
        for(auto& input: tensor->autograd_function->inputs){
            topological_sort(input, visited, topo_order);
        }
    }

    topo_order.push_back(tensor);
}

std::shared_ptr<TensorX> multiply(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y){
    Tensor result = x->get_data() * y->get_data();
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,y,z](){
        Tensor grad_x = z->get_grad() * y->get_data();
        x->accumulate(grad_x);

        Tensor grad_y = z->get_grad() * x->get_data();
        y->accumulate(grad_y);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x, y});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> multiply(std::shared_ptr<TensorX> x, double y){
    Tensor result = x->get_data() * y;
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,y,z](){
        Tensor grad_x = z->get_grad() * y;
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> add(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y){
    Tensor result = x->get_data() + y->get_data();
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,y,z](){
        Tensor grad_x = z->get_grad() * 1;
        x->accumulate(grad_x);

        Tensor grad_y = z->get_grad() * 1;
        y->accumulate(grad_y);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x, y});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> add(std::shared_ptr<TensorX> x, double y){
    Tensor result = x->get_data() + y;
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,z](){
        Tensor grad_x = z->get_grad() * 1;
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> subtract(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y){
    Tensor result = x->get_data() - y->get_data();
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,y,z](){
        Tensor grad_x = z->get_grad() * 1;
        x->accumulate(grad_x);

        Tensor grad_y = z->get_grad() * -1;
        y->accumulate(grad_y);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x, y});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> subtract(std::shared_ptr<TensorX> x, double y){
    Tensor result = x->get_data() - y;
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,z](){
        Tensor grad_x = z->get_grad() * 1;
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> divide(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y){
    Tensor result = x->get_data() / y->get_data();
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,y,z](){
        Tensor grad_x = z->get_grad() / y->get_data();
        x->accumulate(grad_x);

        Tensor y_squared = y->get_data() * y->get_data();
        Tensor grad_y = z->get_grad() * x->get_data() / y_squared * -1.0;
        y->accumulate(grad_y);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x, y});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> divide(std::shared_ptr<TensorX> x, double y){
    Tensor result = x->get_data() / y;
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,y,z](){
        Tensor grad_x = z->get_grad() / y;
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> softmax(std::shared_ptr<TensorX> x, const size_t axis){
    Tensor result = x->get_data().softmax(axis);

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,z, axis](){
        Tensor s = z->get_data();
        Tensor grad_z = z->get_grad();
        Tensor s_prod = (s*grad_z).sum(axis).unsqueeze(axis);
        Tensor s_r = grad_z - s_prod;
        Tensor grad_x = s * s_r;
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    z->set_autograd_fn(autograd);
    return z;
}