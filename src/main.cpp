#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "tensor.h"
#include "nditerator.h"
#include "MatrixMultiply.h"

using namespace std;
using namespace Eigen;

int main(){
    Tensor a({2,3,4});
    Tensor b({4,6});
    size_t c = 0;
    for(size_t i=0; i<2; i++){
        for(size_t j=0; j<3; j++){
            for(size_t k=0; k<4; k++){
                ++c;
                   a.put({i,j,k}, c);
            }
        }
    }

    c = 0;
    //for(size_t i=0; i<2; i++){
        for(size_t j=0; j<4; j++){
            for(size_t k=0; k<6; k++){
                ++c;
                b.put({j,k}, c);
            }
        }
    //}

    a.show();
    cout<<endl;
    b.show();
    cout<<endl;
    Tensor d = MatrixMul::matmul(a,b);
    d.prnt(d.shape());
    d.show();
    cout<<endl;
    return 0;
}