#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <cstddef>

class Optimizer {
    protected:
        double learning_rate;
        size_t time_step;

    public:
        Optimizer(double lr) : learning_rate(lr), time_step(0) {}

        virtual void step();
};

#endif
