//test cases for autograd
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <chrono>
#include "tensor.h"
#include "autograd.h"
#include "tensor_fac.h"
#include "test_utilities.h"

void test_add_tensors(){
    TSUtilities tsutil;
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8,
                                9, 10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23, 24};
    std::vector<size_t> shape = {2, 3, 4};
    std::shared_ptr<TensorX> a = tensor::deep_create(data, shape, true);

    std::shared_ptr<TensorX> z = add(a, a);

    std::vector<double> grad_data = {1, 1, 1, 1, 1, 1, 1, 1, 
                                     1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1};
    Tensor grad_vec(grad_data, shape);
    z->backward(grad_vec);

    std::vector<double> expected_data = {2, 4, 6, 8, 10, 12, 14, 16,
                                         18, 20, 22, 24, 26, 28, 30, 32, 
                                         34, 36, 38, 40, 42, 44, 46, 48};
    std::vector<size_t> expected_shape = {2, 3, 4};
    
    Tensor expected_result(expected_data, expected_shape);
    
    tsutil.test_result("test_add_tensors", tsutil.approximation(z->get_data(), expected_result));
}

void test_add_tensor_double(){
    TSUtilities tsutil;
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8,
                                9, 10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23, 24};
    std::vector<size_t> shape = {2, 3, 4};
    std::shared_ptr<TensorX> a = tensor::deep_create(data, shape, true);

    std::shared_ptr<TensorX> z = add(a, 2);

    std::vector<double> grad_data = {1, 1, 1, 1, 1, 1, 1, 1, 
                                     1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1};
    Tensor grad_vec(grad_data, shape);
    z->backward(grad_vec);

    std::vector<double> expected_data = {3, 4, 5, 6, 7, 8, 9, 10, 
                                         11, 12, 13, 14, 15, 16, 17, 18,
                                         19, 20, 21, 22, 23, 24, 25, 26};
    std::vector<size_t> expected_shape = {2, 3, 4};
    
    Tensor expected_result(expected_data, expected_shape);
    
    tsutil.test_result("test_add_tensor_double", tsutil.approximation(z->get_data(), expected_result));
}

void test_layer_calculation_2d(){
    TSUtilities tsutil;
    std::shared_ptr<TensorX> x = tensor::deep_create({1, 2, 3}, {3, 1}, true);
    std::shared_ptr<TensorX> W = tensor::deep_create({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}, {3, 3}, true);
    std::shared_ptr<TensorX> b = tensor::deep_create({0.1, 0.2, 0.3}, {3, 1}, true);
    x = matmul(W, x);
    x = add(x, b);
    x = relu(x);

    std::shared_ptr<TensorX> l = sum(multiply(x,x), 0);
    l->backward();

    Tensor expected_result({1.5, 3.4, 5.3}, {3,1});

    tsutil.test_result("test_layer_calcuations_2D", tsutil.approximation(x->get_data(), expected_result));
}

void test_layer_calculation_3d(){
    TSUtilities tsutil;
    std::shared_ptr<TensorX> x = tensor::deep_create({1, 2, 3, 4, 5, 6}, {2, 1, 3}, true);
    std::shared_ptr<TensorX> W = tensor::deep_create({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}, {3, 3}, true);
    std::shared_ptr<TensorX> b = tensor::deep_create({0.1, 0.2, 0.3}, {3, 1}, true);
    W = transpose(W);
    std::shared_ptr<TensorX> z = matmul(x, W);

    std::shared_ptr<TensorX> l = sum(multiply(z,z), 0);

    Tensor grad_tensor({1, 1, 1}, {1, 3});
    l->backward(grad_tensor);

    Tensor expected_result({12.2, 69.53, 173.84}, {1, 3});
    Tensor expected_gradient_x({9.84, 11.76, 13.68, 23.88, 28.5, 33.12}, {2,1,3});
    
    tsutil.test_result("test_layer_calcuations_3D", tsutil.approximation(l->get_data(), expected_result));
    tsutil.test_result("test_layer_calculations_3D_gradient", tsutil.approximation(x->get_grad(), expected_gradient_x));
}


int main(){
    test_add_tensor_double();
    test_add_tensors();
    test_layer_calculation_2d();
    test_layer_calculation_3d();
    return 0;
}
