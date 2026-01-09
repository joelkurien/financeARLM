#include "autograd.h"

namespace broadcasts {
    Tensor grad_reshape(Tensor gradient, const std::vector<size_t> target_shape){
        const std::vector<size_t> gradient_shape = gradient.shape();
        for(int i{0}; i<gradient.ndim() && i<target_shape.size(); i++){
            if(gradient_shape[i] > 1 && target_shape[i] == 1) {
                gradient = gradient.sum(i);
            }
        }
        
        return gradient;
    }
}

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
    if(grad_.size() == 0)
        grad_ = new_grad;
    else{
        grad_ = grad_ + new_grad;
    }
}        

void TensorX::grad_zeros() {
    if(required_grad)
        grad_ = Tensor(data_.shape());
}

void TensorX::backward(const std::optional<Tensor>& grad) {
    if(!required_grad) throw std::runtime_error("Gradient requirement was set to false");

    //initialize our gradient
    if(grad.has_value()){
        if(grad->shape() != data_.shape()){
            throw std::runtime_error("Gradient/Data shape mismatch");
        }
        std::vector<double> grad_values(grad->size(), 1);
        grad_ = Tensor(grad_values, grad->shape());
    }
    else {
        if(data_.size() != 1){
            throw std::runtime_error("Gradient is implictly set to scalar");
        }
        grad_ = Tensor({1.0}, data_.shape());
    }

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

void TensorX::topological_sort(std::shared_ptr<TensorX> tensor, std::unordered_set<TensorX*>& visited, std::vector<std::shared_ptr<TensorX>>& topo_order) {
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
        Tensor grad_x = broadcasts::grad_reshape(z->get_grad() * y->get_data(), x->get_data().shape());
        Tensor grad_y = broadcasts::grad_reshape(z->get_grad() * x->get_data(), y->get_data().shape());

        x->accumulate(grad_x);
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
        Tensor grad_x = broadcasts::grad_reshape(z->get_grad(), x->get_data().shape());
        Tensor grad_y = broadcasts::grad_reshape(z->get_grad(), y->get_data().shape());

        x->accumulate(grad_x);
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
        Tensor grad_x = broadcasts::grad_reshape(z->get_grad()*1, x->get_data().shape());
        Tensor grad_y = broadcasts::grad_reshape(z->get_grad()*-1, y->get_data().shape());

        x->accumulate(grad_x);
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
        Tensor grad_x = broadcasts::grad_reshape(z->get_grad() / y->get_data(), x->get_data().shape());

        Tensor y_squared = y->get_data() * y->get_data();
        Tensor grad_y = broadcasts::grad_reshape(z->get_grad() * x->get_data() / y_squared * -1.0, y->get_data().shape());
        
        x->accumulate(grad_x);
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
        Tensor s_prod = (s*grad_z).sum(axis);
        Tensor s_r = grad_z - s_prod;
        Tensor grad_x = s * s_r;
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> log_softmax(std::shared_ptr<TensorX> x, const size_t axis){
    Tensor result = x->get_data().log_softmax(axis);

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,z, axis](){
        Tensor s = z->get_data();
        Tensor grad_z = z->get_grad();
        Tensor grad_sum = (grad_z).sum(axis);
        Tensor grad_x = grad_z - s * grad_sum;
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> layer_norm(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> gamma, std::shared_ptr<TensorX> beta, const size_t axis){
    std::shared_ptr<TensorX> mean_of_x = mean(x, axis);
    std::shared_ptr<TensorX> centered = subtract(x, mean_of_x);
    std::shared_ptr<TensorX> squared = pow(centered, 2);
    std::shared_ptr<TensorX> variance = mean(squared, axis);

    double e = 1e-12;
    std::shared_ptr<TensorX> var = add(variance, e);
    std::shared_ptr<TensorX> std = sqrt(var);
    std::shared_ptr<TensorX> lnorm = divide(centered, std);
    std::shared_ptr<TensorX> mul = multiply(gamma, lnorm);
    std::shared_ptr<TensorX> res = add(mul, beta);
    return res;
}


std::shared_ptr<TensorX> sum(std::shared_ptr<TensorX> x, const size_t axis){
    Tensor result = x->get_data().sum(axis);
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,z, axis](){
        Tensor grad_z = z->get_grad();
        Tensor res = grad_z.expand(x->get_data().shape());
        x->accumulate(res);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> mean(std::shared_ptr<TensorX> x, const size_t axis){
    Tensor result = x->get_data().mean(axis);

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,z, axis](){
        Tensor grad_z = z->get_grad() * (1.0/x->get_data().shape()[axis]);
        Tensor res = grad_z.expand(x->get_data().shape());
        x->accumulate(res);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> var(std::shared_ptr<TensorX> x, const size_t axis){
    std::shared_ptr<TensorX> mean_of_x = mean(x, axis);
    std::shared_ptr<TensorX> centered = subtract(x, mean_of_x);
    std::shared_ptr<TensorX> squared = pow(centered, 2);
    std::shared_ptr<TensorX> variance = mean(squared, axis);
    return variance;
}

std::shared_ptr<TensorX> relu(std::shared_ptr<TensorX> x){
    Tensor result = x->get_data().relu();

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,z, result](){
        Tensor grad_z = z->get_grad();

        std::vector<double> mask(result.size(), 0);
        const double* resPtr = result.data();

        #pragma omp parallel for simd schedule(static)
        for(size_t i=0; i<result.size(); i++){
            mask[i] = resPtr[i] > 0.0 ? 1.0 : 0.0;
        }

        Tensor masked(mask, result.shape());
        Tensor res = grad_z * masked;
        x->accumulate(res);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    z->set_autograd_fn(autograd);
    return z;
}

static double gelu_derivative(double x){
    const double gamma = 0.044715;
    const double sqrt_over_pi = sqrt(2.0 / std::numbers::pi);
    const double cube = x*x*x;
    const double inner = sqrt_over_pi*(x+gamma*cube);
    const double sech_squared = 1-tanh(inner)*tanh(inner);
    const double pdf = 0.5*x*sech_squared;
    const double cdf = 0.5*(1.0+tanh(inner));
    return pdf+cdf;
}

std::shared_ptr<TensorX> gelu(std::shared_ptr<TensorX> x){
    Tensor result = x->get_data().gelu();

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z, result](){
        Tensor grad_z = z->get_grad();

        std::vector<double> grad_values(result.size(), 0);
        const double* resPtr = result.data();
        
        #pragma omp parallel for simd schedule(static)
        for(size_t i=0; i<result.size(); i++){
            grad_values[i] = gelu_derivative(resPtr[i]);
        } 

        Tensor grad(grad_values, result.shape());
        Tensor res = grad_z * grad;
        x->accumulate(res);       
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> matmul(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y){
    Tensor result = MatrixMul::matmul(x->get_data(), y->get_data());
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, y, z] {
        Tensor grad_z = z->get_grad();
        Tensor x_grad = MatrixMul::matmul(grad_z, y->get_data().transpose());
        Tensor y_grad = MatrixMul::matmul(x->get_data().transpose(), grad_z);
        x->accumulate(x_grad);
        y->accumulate(y_grad);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x,y});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> transpose(std::shared_ptr<TensorX> x){
    Tensor result = x->get_data().transpose();
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z] {
        Tensor grad_z = z->get_grad();
        Tensor grad_T = grad_z.transpose();
        x->accumulate(grad_T);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> permute(std::shared_ptr<TensorX> x, const std::optional<std::vector<size_t>>& rotaxis){
    std::vector<size_t> og_rotaxis(x->get_data().ndim()); 
    std::iota(og_rotaxis.begin(), og_rotaxis.end(), 0);
    Tensor result = x->get_data().permute(rotaxis);
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z, og_rotaxis] {
        Tensor grad_z = z->get_grad();
        Tensor grad_x = grad_z.permute(og_rotaxis);
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> reshape(std::shared_ptr<TensorX> x, std::vector<size_t> new_shape){
    std::vector<size_t> old_shape = x->get_data().shape(); 
    Tensor result = x->get_data().reshape(new_shape); 

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z, old_shape] {
        Tensor grad_z = z->get_grad();
        Tensor grad_x = grad_z.reshape(old_shape);
        x->accumulate(grad_x);
    };
    
    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> concat(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y, const size_t axis){
    Tensor result = x->get_data().concatenate(y->get_data(), axis);   

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, y, z] {
        Tensor x_data = x->get_data();
        Tensor y_data = y->get_data();

        Tensor grad_z = z->get_grad();
        Tensor grad_x = grad_z.slice(0, x_data.size(), x_data.shape());
        Tensor grad_y = grad_z.slice(x_data.size(), grad_z.size(), y_data.shape());
        x->accumulate(grad_x);
        y->accumulate(grad_y);
    };
    
    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x,y});
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> sqrt(std::shared_ptr<TensorX> x){
   Tensor result = x->get_data().sqrt();
   std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

   auto backward_fn = [x, z] {
       Tensor grad_z = z->get_grad();
       Tensor grad_x = grad_z / (z->get_data() * 2);

       x->accumulate(grad_x);
   };

   std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
   z->set_autograd_fn(autograd);
   return z;
}

std::shared_ptr<TensorX> exp(std::shared_ptr<TensorX> x){
   Tensor result = x->get_data().exp();
   std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

   auto backward_fn = [x, z] {
       Tensor grad_z = z->get_grad();
       Tensor grad_x = grad_z * z->get_data();
       x->accumulate(grad_x);
   };

   std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
   z->set_autograd_fn(autograd);
   return z;
}

std::shared_ptr<TensorX> log(std::shared_ptr<TensorX> x){
   Tensor result = x->get_data().log();
   std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

   auto backward_fn = [x, z] {
       Tensor grad_z = z->get_grad();
       Tensor grad_x = grad_z / x->get_data();
       x->accumulate(grad_x);
   };

   std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
   z->set_autograd_fn(autograd);
   return z;
}

std::shared_ptr<TensorX> pow(std::shared_ptr<TensorX> x, const double n){
   Tensor result = x->get_data().pow(n);
   std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

   auto backward_fn = [x, z, n] {
       Tensor grad_z = z->get_grad();
       Tensor grad_x = (grad_z)*(x->get_data().pow(n-1) * n);

       x->accumulate(grad_x);
   };

   std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
   z->set_autograd_fn(autograd);
   return z;

}






