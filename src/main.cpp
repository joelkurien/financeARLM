#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "tensor.h"
#include "nditerator.h"
#include "MatrixMultiply.h"
#include "autograd.h"

using namespace std;
using namespace Eigen;

int main(){
    Tensor a({3,3,4});
    Tensor b({2,3,4});
    size_t c = 0;
    for(size_t i=0; i<3; i++){
        for(size_t j=0; j<3; j++){
            for(size_t k=0; k<4; k++){
                ++c;
                   a.put({i,j,k}, c);
            }
        }
    }

    c = 0;
    for(size_t i=0; i<2; i++){
        for(size_t j=0; j<3; j++){
            for(size_t k=0; k<4; k++){
                ++c;
                b.put({i,j,k}, c);
            }
        }
    }

    // Tensor t = a.softmax(1);
    // t.prnt(t.shape());
    // t.show();

    // vector<size_t> bob = {2};
    // auto indices = NDRange(bob);
    // for(auto& idx: indices){
    //     cout<<idx[0]<<endl;
    // }
    
    // a.show();
    // cout<<endl;
    // b.show();
    // cout<<endl;
    // Tensor d = a.concatenate(b,0);
    // for(size_t i=0; i<5; i++){
    //     for(size_t j=0; j<3; j++){
    //         for(size_t k=0; k<4; k++){
    //             cout<<d.at({i,j,k})<<" ";
    //         }
    //         cout<<endl;
    //     }
    //     cout<<endl;
    // }
    // d.prnt(d.shape());
    // d.show();
    cout<<endl;
    vector<bool> mask(36, false);
    for(size_t i = 0; i < 36; i++) {
        mask[i] = (static_cast<int>(a.as_vector()[i]) % 2 == 1);  // true for odd numbers
    }

    // Apply mask - replace all odd numbers with 0
    // Tensor result = a.mask_filled(mask, 0.0);
    // result.prntd(result.as_vector());
    Tensor res = a.gelu();
    res.prntd(res.as_vector());
    return 0;
}