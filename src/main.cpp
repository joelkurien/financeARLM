#include <vector>
#include "tensor.h"
#include "autograd.h"
#include "tensor_fac.h"

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

    std::shared_ptr<TensorX> a = tensor::deep_create({5,4}, true);
    std::shared_ptr<TensorX> c = tensor::deep_create({5,4}, true);

    int lol = 0;
    bool flag = false;
    for(size_t i=0; i<5; i++){
        for(size_t j=0; j<4; j++){
            a->get_data().put({i,j}, -lol++);
        }
    }

    int d=0;
    for(size_t i=0; i<5; i++){
        for(size_t j=0; j<4; j++){
            c->get_data().put({i,j}, d++*2);
        }
    }

    std::vector<Tensor> con = a->get_data().chunk(3, 1);
    Tensor sig = a->get_data().sigmoid();
    Tensor th = a->get_data().tanh();
    sig.prntd(sig.as_vector_const());
    th.prntd(th.as_vector_const());

    std::cout<<"Conc vector ";
    Tensor m = con[1].mean(1);
    m.prntd(con[1].as_vector_const());
    m.prntd(m.as_vector_const());
    m.prnt(m.shape());
    std::cout<<"Conc shape ";
    std::cout<<con.size()<<std::endl;

    a->get_data() += c->get_data();
    m.prntd(a->get_data().as_vector_const());
    //con[0].prnt(con[2].shape());
    // std::shared_ptr<TensorX> gamma = tensor::deep_create({1,4}, true);
    // std::shared_ptr<TensorX> beta  = tensor::deep_create({1,4}, true);
    //
    //
    // for (size_t j = 0; j < 4; ++j) {
    //     gamma->get_data().put({0,j}, 1.0);  
    //     beta->get_data().put({0,j}, 0.0);   
    // }
    //
    
    std::vector<double> mask_data = {
        1, 1, 1, 1,  // Row 0: Masked
        0, 0, 0, 0,  // Row 1: Kept
        0, 0, 0, 0   // Row 2: Kept
    };
    Tensor mask(mask_data, a->get_data().shape());

    // 3. Forward Pass
    double replace_val = -1e9;
    std::vector<size_t> p = {1,0};
    Tensor r = a->get_data();
    Tensor f = r.maximum(1);
    Tensor msk = r == f;
    r.prntd(r.as_vector_const());
    f.prntd(f.as_vector_const());
    std::cout<<"Mask of r and f ";
    msk.prntd(msk.as_vector_const());


    std::shared_ptr<TensorX> r_ = tensor::create(r, true);
    std::shared_ptr<TensorX> z = divide(r_, c);
    std::cout<<"Concatenated matrix: ";
    z->get_data().prntd(z->get_data().as_vector());
    std::shared_ptr<TensorX> b = sum(z, 1);
    std::shared_ptr<TensorX> l = sum(b,0);
    l->backward();
    //l->get_data().prntd(l->get_data().as_vector());
    std::cout<<"z grad output"<<std::endl; 
    l->get_data().prntd(z->get_grad().as_vector());

    std::cout<<"b data output"<<std::endl;
    b->get_data().prntd(b->get_data().as_vector());

    std::cout<<"l data output"<<std::endl;
    l->get_data().prntd(l->get_data().as_vector());

    std::cout<<"a grad output"<<std::endl;
    l->get_data().prntd(a->get_grad().as_vector());

    return 0;
}
