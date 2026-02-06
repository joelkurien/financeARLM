#ifndef ADAMW_H
#define ADAMW_H

#include "Optimizer.h"
#include "autograd.h"
#include "tensor.h"
#include "tensor_fac.h"

class AdamW : public Optimizer {
    double beta1;
    double beta2;
    double epsilon;
    double lambda;
    std::vector<Tensor> momentum;
    std::vector<Tensor> velocity;
    std::vector<std::shared_ptr<TensorX>> parameters; 

    public:
        AdamW(std::vector<std::shared_ptr<TensorX>> params, 
                double lr, double b1, 
                double b2, double e,
                double l)
            : Optimizer(lr), 
              parameters(params),
              beta1(b1),
              beta2(b2),
              epsilon(e),
              lambda(l)
        {
            for(std::shared_ptr<TensorX>& param: parameters){
                momentum.push_back(Tensor(param->get_data().shape()));
                velocity.push_back(Tensor(param->get_data().shape()));
            }
        }

        virtual void step() override {
            time_step++;
            for(size_t i=0; i<parameters.size(); i++){
                std::shared_ptr<TensorX>& parameter = parameters[i];
                std::cout<<parameter->get_required_grad()<<std::endl;
                if(!parameter->get_required_grad()) continue;
                Tensor& param = parameter->get_data();
                Tensor gradient = parameter->get_grad();
                Tensor& m = momentum[i];
                Tensor& v = velocity[i];
                
                std::cout<<"COB"<<std::endl;
                prntd(param.as_vector_const());
                prntd(m.as_vector_const());
                prntd(gradient.as_vector_const());
                m = m*beta1 + gradient*(1-beta1);
                std::cout<<"MOB"<<std::endl;
                v = v*beta2 + (gradient * gradient) * (1-beta2);
                
                std::cout<<"FOB"<<std::endl;
                double m_bias_correction = (1-std::pow(beta1,time_step));
                double v_bias_correction = (1-std::pow(beta2,time_step));
                
                Tensor mt_hat = m / m_bias_correction;
                Tensor vt_hat = v / v_bias_correction;
                
                std::cout<<"BOB"<<std::endl;
                //Regularization
                param -= (param * lambda);

                //Gradient loss
                Tensor cost_fn_grad = mt_hat / (vt_hat.sqrt() + epsilon);
                param -= cost_fn_grad * learning_rate; 
            }
        }

        void zero_grad(){
            for(std::shared_ptr<TensorX>& params: parameters){
                params->grad_zeros();
            }
        }
};

#endif
