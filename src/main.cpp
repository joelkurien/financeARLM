#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "tensor.h"
#include "nditerator.h"

using namespace std;
using namespace Eigen;

int main(){
    Tensor a({3,4});
    Tensor b({2,3,4});
    double c = 0;
    for(size_t i=0; i<3; i++){
        for(size_t j=0; j<4; j++){
                a.put({i,j}, c++);
        }
    }
    double d = 0;
    for(size_t i=0; i<2; i++)
    for(size_t j=0; j<3; j++)
    for(size_t k=0; k<4; k++){
        b.put({i,j, k}, d++);
    }
    Tensor x = a+b;
    for(size_t i=0; i<2; i++)
    for(size_t j=0; j<3; j++){
        for(size_t k=0; k<4; k++){
            cout<<x.at({i,j,k})<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
    Tensor y = x.layer_norm(1,1,1);
    y.prnt(y.shape());
    for(size_t i=0; i<2; i++)
    for(size_t j=0; j<3; j++){
        for(size_t k=0; k<4; k++){
            cout<<y.at({i,j,k})<<" ";
        }
        cout<<endl;
    }
    // cout<<a.shape_check(b.shape())<<endl;
    return 0;
}