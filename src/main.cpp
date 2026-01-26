#include <vector>
#include "tensor.h"
#include "autograd.h"
#include "tensor_fac.h"
#include "grn.h"

int main(){
    Tensor data = arange({10,5});
    std::shared_ptr<TensorX> data_tensor = tensor::create(data);

    GRN grn_net(256);

    std::shared_ptr<TensorX> grn_res = grn_net.forward(data_tensor);

    for(size_t i=0; i<10; i++){
        for(size_t j=0; j<5; j++){
            std::cout<<grn_res->get_data().at({i,j})<<" ";
        }
        std::cout<<std::endl;
    }
    return 0;
}
