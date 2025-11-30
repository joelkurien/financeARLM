#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "tensor.h"
#include "nditerator.h"

using namespace std;
using namespace Eigen;

int main(){
    Tensor a({2,2});
    Tensor b({2,2});
    double c = 0;
    for(size_t i=0; i<2; i++){
        for(size_t j=0; j<2; j++){
                a.put({i,j}, c++);
        }
    }
    double d = 0;
    for(size_t i=0; i<2; i++)
    for(size_t j=0; j<2; j++){
        b.put({i,j}, d+=2);
    }
    Tensor x = a*2.695784;
    x.prnt(x.shape());
    for(size_t i=0; i<2; i++){
        for(size_t j=0; j<2; j++){
            cout<<x.at({i,j})<<" ";
        }
        cout<<endl;
    }
    return 0;
}