#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "tensor.h"

using namespace std;
using namespace Eigen;

int main(){
    Tensor tensor({2,4});
    tensor.show();
    cout<<endl;
    // vector<size_t> t_shape = tensor.shape();
    // for(auto sh: t_shape){
    //     cout<<sh<<" ";
    // } 
    // cout<<endl;
    size_t c = 0;
    for(size_t i=0; i<2; i++){
        for(size_t j=0; j<4; j++){
            tensor.put({i,j}, c++);
        }
    }
    tensor.show();
    Tensor reshaped = tensor.reshape({4,2});
    Tensor permutate = tensor.permute();
    Tensor slc = reshaped.slice({1,0}, {2,2}).permute();
    cout<<"Tranpose: "<<endl;
    for(size_t i=0; i<4; i++){
        for(size_t j=0; j<2; j++){
            cout<<permutate.at({i,j})<<" ";
        }
        cout<<endl;
    }
    cout<<"Reshaped: "<<endl;
    for(size_t i=0; i<4; i++){
        for(size_t j=0; j<2; j++){
            cout<<reshaped.at({i,j})<<" ";
        }
        cout<<endl;
    }
    slc.show();
    cout<<"Sliced: "<<endl;
    for(size_t i=0; i<2; i++){
        for(size_t j=0; j<2; j++){
            cout<<slc.at({i,j})<<" ";
        }
        cout<<endl;
    }
    return 0;
}