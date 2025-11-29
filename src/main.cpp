#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "tensor.h"

using namespace std;
using namespace Eigen;

int main(){
    Tensor a({3,4});
    Tensor b({2,3,4});
    double c = 0;
    //for(size_t i=0; i<2; i++){
        for(size_t j=0; j<3; j++){
            for(size_t k=0; k<4; k++){
                a.put({j,k}, c++);
            }
        }
    //}
    double d = 0;
    for(size_t i=0; i<2; i++)
    for(size_t j=0; j<3; j++){
        for(size_t k=0; k<4; k++){
            b.put({i,j,k}, d++);
        }
    }
    Tensor x = a+b;
    
    cout<<endl;
    return 0;
}