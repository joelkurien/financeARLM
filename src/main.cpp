#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "tensor.h"

using namespace std;
using namespace Eigen;

int main(){
    Tensor tensor(2,2);
    auto [row, col] = tensor.shape();
    tensor.put(1,1,9);
    cout<<tensor[1][1]<<endl;
    cout<<tensor.empty()<<endl;
    cout<<tensor.size()<<endl;
    return 0;
}