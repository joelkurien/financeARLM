#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "tensor.h"

using namespace std;
using namespace Eigen;

int main(){
    Tensor tensor(3,3);
    auto [row, col] = tensor.shape();
    int s = 0;
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            tensor.put(i,j,s++);
        }
    }

    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            cout<<tensor[i][j]<<" ";
        }
        cout<<endl;
    }
    auto [smaddr, height, width] = tensor.view(1, 2, 1, 2);
    Tensor slc = tensor.slice(1,1,2,2);
    auto [sr, sc] = slc.shape();
    for(int i=0; i<sr; i++){
        for(int j=0; j<sc; j++){
            cout<<slc[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
    return 0;
}