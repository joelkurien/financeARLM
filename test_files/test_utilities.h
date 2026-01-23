#include<iostream>
#include "tensor.h"

class TSUtilities {
    public: 
        bool approximation(const Tensor& result, const Tensor& target, const double thres = 1e-6){
            if(result.shape() != target.shape()) {
                std::cout<<"Observed tensor shape mismatches with target tensor shape : " + vec_string(result.shape()) + " v " + vec_string(target.shape())<<std::endl;
                return false;
            }

            for(size_t i=0; i<result.size(); i++){
                if(std::abs(result.data()[i] - target.data()[i]) > thres){
                    std::cout<<"Mismatch in observed values of index (" << i << ") \n Observed value: " << result.data()[i] << " \n Target Value: " << target.data()[i]<<std::endl;
                    return false;
                }
            }
            return true;
        }

        void test_result(const std::string& test_name, const bool passed, double time = 0.0){
            std::string isPassed = passed ? "true" : "false";
            std::cout<<test_name << ": " <<passed <<" Time Taken: "<< time <<" ms"<<std::endl;
        } 
};
