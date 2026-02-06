#ifndef LRSCHEDULER_H
#define LRSCHEDULER_H

#include <cstddef>

class LRScheduler {
    protected:
        double base_learning_rate;
        size_t current_step;

    public:
        LRScheduler(double base_lr)
            : base_learning_rate(base_lr), current_step(0) {}
        
        virtual double get_learning_rate() = 0;

        void step() {
            current_step++;
        }

        void reset() { current_step = 0; }

        size_t get_current_step() const { return current_step; }

        virtual ~LRScheduler() = default;
};

#endif
