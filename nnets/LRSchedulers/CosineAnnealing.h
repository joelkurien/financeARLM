#ifndef COSINEANNEAL_H
#define COSINEANNEAL_H

#include <string>
#include <stdexcept>
#include <cmath>
#include "LRScheduler.h"

class CosineAnnealing : public LRScheduler{
    double min_learning_rate;
    size_t total_steps;
    size_t warmup_steps;

    public:
        CosineAnnealing(double base_lr,
                        size_t t_steps, 
                        size_t w_steps, 
                        double min_lr = 0.0)
            : LRScheduler(base_lr), 
              total_steps(t_steps), 
              warmup_steps(w_steps), 
              min_learning_rate(min_lr)
        {
            if(warmup_steps > total_steps) 
                throw std::runtime_error("The warmup steps should be lower than the total number of steps" 
                        + std::to_string(total_steps) + ", " + std::to_string(warmup_steps));
        }

        virtual double get_learning_rate() override {
            if(current_step < warmup_steps){
                if(warmup_steps == 0)
                    return base_learning_rate;
                return base_learning_rate * (static_cast<double>(current_step)/warmup_steps); 
            }

            if(current_step < total_steps){
                size_t decay_steps = total_steps - warmup_steps;
                size_t from_warmup = current_step - warmup_steps;

                double current_status = (static_cast<double>(from_warmup)/decay_steps); 
                double cos_decay = 0.5 * (1 + std::cos(M_PI * current_status));

                return min_learning_rate + (base_learning_rate - min_learning_rate) * cos_decay;
            }

            return min_learning_rate;
        }
};

#endif
