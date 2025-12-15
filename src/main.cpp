#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "tensor.h"
#include "nditerator.h"
#include "MatrixMultiply.h"
#include "autograd.h"
#include "tensor_fac.h"

using namespace std;
using namespace tensor;

int main(){
    std::shared_ptr<TensorX> a = tensor::deep_create({3,1}, true);
    std::shared_ptr<TensorX> b = tensor::deep_create({3,1}, true);
    for(size_t x=0; x<3; x++){
        a->get_data().put({x,0},x);
    }
    for(size_t x=0; x<3; x++){
        b->get_data().put({x,0}, x+2);
    }
    a->get_data().prntd(a->get_data().as_vector());
    b->get_data().prntd(b->get_data().as_vector());
    std::shared_ptr c = multiply(a,b);
    c->get_data().prntd(c->get_data().as_vector());
    c->backward();
    a->get_grad().prntd(a->get_grad().as_vector());
    b->get_grad().prntd(b->get_grad().as_vector());
    return 0;
}