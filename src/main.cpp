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
    // std::shared_ptr<TensorX> a = tensor::deep_create({3,1}, true);
    // std::shared_ptr<TensorX> b = tensor::deep_create({3,1}, true);
    // for(size_t x=0; x<3; x++){
    //     a->get_data().put({x,0},x);
    // }
    // for(size_t x=0; x<3; x++){
    //     b->get_data().put({x,0}, x+2);
    // }
    // a->get_data().prntd(a->get_data().as_vector());
    // b->get_data().prntd(b->get_data().as_vector());
    // std::shared_ptr c = divide(a,4);
    // c->get_data().prntd(c->get_data().as_vector());
    // c->backward();
    // c->backward();
    // a->get_grad().prntd(a->get_grad().as_vector());
    // std::shared_ptr<TensorX> d = softmax(a,0);
    // d->backward();
    // a->get_grad().prntd(a->get_grad().as_vector());
    //b->get_grad().prntd(b->get_grad().as_vector());

    // Tensor x({3,1});
    // Tensor y({3,1});
    // double j=0;
    // double k=0;
    // for(size_t i=0; i<3; i++){
    //     x.put({i,0}, i);
    //     y.put({i,0}, i);
    // }

    // x.prntd(x.as_vector());
    // y.prntd(y.as_vector());
    // Tensor z = dot(x,y,0);
    // z.prntd(z.as_vector());

    std::shared_ptr<TensorX> a = tensor::deep_create({3,4}, true);
    std::shared_ptr<TensorX> c = tensor::deep_create({5,4}, true);

    int lol = 0;
    bool flag = false;
    for(size_t i=0; i<3; i++){
        for(size_t j=0; j<4; j++){
            a->get_data().put({i,j}, lol++);
        }
    }

    int d=0;
    for(size_t i=0; i<5; i++){
        for(size_t j=0; j<4; j++){
            c->get_data().put({i,j}, d++*2);
        }
    }

    std::shared_ptr<TensorX> z = concat(a,c,0);
    std::shared_ptr<TensorX> b = sum(z, 1);
    std::shared_ptr<TensorX> l = sum(b,0);
    l->backward();
    //l->get_data().prntd(l->get_data().as_vector());
    l->get_data().prnt(z->get_data().shape());
    //l->get_data().prntd(z->get_data().as_vector());
    l->get_data().prntd(a->get_grad().as_vector());
    l->get_data().prnt(c->get_grad().shape());
    return 0;
}
