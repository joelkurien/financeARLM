#ifndef UTILITY_H
#define UTILITY_H

#include<cmath>
#include<iostream>
#include<vector>
#include "tensor.h"

class Utilities {
    
    public:
        Tensor grad_reshape(Tensor gradient, const std::vector<size_t> target_shape){
            const std::vector<size_t> gradient_shape = gradient.shape();
            Tensor result = gradient;
            for(int i{0}; i<gradient.ndim() && i<target_shape.size(); i++){
                if(gradient_shape[i] == 1 && target_shape[i] > 1) {
                    result = result.sum(i);
                }
            }
            
            return result;
        }
};


#endif
