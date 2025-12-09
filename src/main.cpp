#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "tensor.h"
#include "nditerator.h"
#include "MatrixMultiply.h"

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

    vector<size_t> bob = {2};
    auto indices = NDRange(bob);
    for(auto& idx: indices){
        cout<<idx[0]<<endl;
    }

    // a.show();
    // cout<<endl;
    // b.show();
    // cout<<endl;
    Tensor d = a.concatenate(b,0);
    for(size_t i=0; i<5; i++){
        for(size_t j=0; j<3; j++){
            for(size_t k=0; k<4; k++){
                cout<<d.at({i,j,k})<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
    d.prnt(d.shape());
    d.show();
    cout<<endl;
    return 0;
}