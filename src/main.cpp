#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "tensor.h"
#include "nditerator.h"

using namespace std;
using namespace Eigen;

int main(){
    Tensor a({4,3});
    Tensor b({2,4,3});
    double c = 0;
    for(size_t i=0; i<4; i++){
        for(size_t j=0; j<3; j++){
                a.put({i,j}, c++);
        }
    }
    double d = 1;
    for(size_t i=0; i<2; i++)
    for(size_t j=0; j<4; j++)
    for(size_t k=0; k<3; k++){
        b.put({i,j, k}, d++);
    }
    Tensor x = a+b;
    x.prnt(x.shape());
    for(size_t i=0; i<2; i++)
    for(size_t j=0; j<4; j++){
        for(size_t k=0; k<3; k++){
            cout<<x.at({i,j,k})<<" ";
        }
        cout<<endl;
    }
    cout<<a.shape_check(b.shape())<<endl;
    return 0;
}