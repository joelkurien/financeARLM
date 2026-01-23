#ifndef ROOTLAYER_H
#define ROOTLAYER_H

#include "autograd.h"

class RootLayer {
    public: 
        virtual ~RootLayer() = default; //allows for children to handle memory leaks their own way rather than forcing the parents destructor every time
        
        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input); 
        
        virtual std::vector<std::shared_ptr<TensorX>> parameters() = 0; //stores the list of all parameters as a vector for each layer
};

#endif
