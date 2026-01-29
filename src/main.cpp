#include <vector>
#include "tensor.h"
#include "autograd.h"
#include "tensor_fac.h"
#include "vsn.h"
#include "static_encoder.h"

int main(){
    Tensor data = arange({10,5});
    std::shared_ptr<TensorX> data_tensor = tensor::create(data);

    Static_Encoder se(256, 0.3, 5);

    std::shared_ptr<TensorX> vsn_res = se.forward(data_tensor);

    // for(size_t i=0; i<2; i++){
    //     for(size_t j=0; j<10; j++){
    //         for(size_t k=0; k<5; k++){
    //             std::cout<<vsn_res->get_data().at({i,j,k})<<" ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }
    for(size_t i=0; i<10; i++){
        for(size_t j=0; j<5; j++){
            std::cout<<vsn_res->get_data().at({i,j})<<" ";
        }
        std::cout<<std::endl;
    }

    prnt(vsn_res->get_data().shape());

    return 0;
}
