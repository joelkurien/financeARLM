#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "tensor.h"

using namespace std;
using namespace Eigen;

int main(){
    Tensor tensor({4,2});
    vector<size_t> t_shape = tensor.shape();
    for(auto sh: t_shape){
        cout<<sh<<" ";
    } 
    cout<<endl;
    for(size_t i=0; i<tensor.size(); i++){
        tensor.put(i);
    }

    Tensor reshaped = tensor.reshape({2,4});
    Tensor slc = reshaped.slice({1,1}, {1,3});
    for(size_t i=0; i<1; i++){
        for(size_t j=0; j<3; j++){
            cout<<slc.at({i,j})<<" ";
        }
        cout<<endl;
    }
    return 0;
}