# Financial ARLM: Transformer From Scratch + Backtesting Engine

This project implements a transformer architecture entirely from scratch and uses it to build an Autonomous Research Language Model (ARLM) capable of summarizing financial documents and extracting key signals for downstream trading strategies. A custom backtesting engine is being developed alongside the ARLM for strategy evaluation.

## Tensor Library (C++)

**Day 1 — Build & Linear Algebra Setup**  
Created the C++ executable framework with CMake. Integrated the Eigen library to support matrix operations and high-performance linear algebra routines.

**Day 2 — Initial Tensor System (2D)**  
Implemented a basic tensor class focused on 2D operations. Added shape handling, update functions, and stride-based memory views to support efficient in-place matrix updates.

**Day 3 — Batched 3D Tensors**  
Extended the tensor class to 3D for batch processing (essential for transformer workloads). Added batch slicing, deep/shallow views, and batched update operations.

**Day 4 — Unified General Tensor Class**  
Merged the 2D and 3D structures into a single N-dimensional tensor using a raw pointer + stride arithmetic for data access. Added flexible slicing, reshaping, and optimized stride computation for contiguous and non-contiguous views.

**Day 5 — Broadcasting Support**  
Implemented complete broadcasting rules: shape inference, leading/trailing singleton handling, stride rewriting, and unsqueeze/expand mechanics. Added scaffolding for element-wise arithmetic using stride-based iteration.

**Day 6 — Transformer Arithematic Support**  
Implemented arithematic methods that a transformer needs: Matrix addition, linear matrix multiplication, scalar matrix multiplication, scalar matrix division, scalar matrix subtraction.

**Day 7 — Refactoring and Reductions**  
Refactored the element-wise addition, subtraction, division and multiplication, also implemented the reduction across single axes logic - will apply it for sum, mean, min and min later. Also generalized broadcasting for each operation. Also added parallelization to computation intensive operations

**Day 8 - Reductions and functional operations**    
Implemented the sum, mean and maximum finding along an axis, also completed the implementation of softmax along an axis and layernormalization along an axis with a parameterized gamma and beta values that can be passed and a fixed error variance value.

**Day 9 - Matrix multiplication and broadcasting rules**    
Implemented matrix multiplication using BLAS library because it was too complex and time consuming. I had to refer multiple sources on how to handle concurrency and control GPU/CPU performance so that I dont burn out my CPU, but failed miserably so I just resorted to using BLAS because I am lazy af.

**Day 10 - Broadcasting rules** 
Completed all required broadcasting rules that are necessary to build a transformer/basic feedforward network. Further implemented the Relu and Gelu functions as well. We have completed all the necessary functions required by a tensor library to build transformers and neural networks. Fucking hell!!!! I fucking hate C++ but have to do it if I want to get a good career somewhere. 

## Autograd Library (C++)

**Day 1- Computational graph**  
Why the fuck did I take up this mountanous task, creating the computational graph itself is hell, if Jane Street or some other fucking company does not hire me, I will fucking burn someone's house down. Fucking cunt ass job market. Anyways, created the computational graph, had to learn topological graphs, shared pointers, smart pointers and memory management for the same. I had to refer to code from other sources since I could not understand what was going on in this shit, and it took about 4hrs to write like 20-30 lines of code. I just want to die OMG.

**Day 2- Elementwise operations gradients** 
Cleaned up my code and separated my .h and .cpp files. Created autograd operations for elementwise arithematic operations

**Day 3- Softmax gradients and debugging**
Today was intended to implement the gradients for all the reduction operations but it was cut short due to the issue of debugging and resolving issues that was caused due to the unsqueeze() function. The cause of issue - when unsqueeze was called on a temporary object that was bound to be destroyed once called (sum(axis) that returns a tensor) the base pointer of the unsqueeze will then be pointing to jackshit => dangling pointer, due to this issue not being found I wasted about 3 hrs, learning about dangling pointers and started to get annoyed, so I just made the unsqueeze to create a brand new tensor instead of the originally intended shallow copy. I had taken assistance from AI sources to understand how to resolve it and they had recommended using shared_ptr, but as I am unaware of how to use shared_ptr, we will have to overhaul and remodel the tensor library in a later date to incorporate shared_ptr and unique_ptr for dereferencing safety. For now, we have completed the implementation of softmax gradients, tmrw will look forward to completing the other reductions and may be, just may be do matrix multiplication as well - if it poses to be hell I will be using libraries.

**Day 4- Reduction operation and matrix multiplication gradients**  
Implemented the backward functions for mean, sum along an axis, relu and gelu functions, matrix multiplication, tranpose and permute operations. Learned more about dereferencing and viewing an instance from another one.

**Day 5- Simple Matrix operations autograd**
Implemented the autograd for reshaping and concatenating a matrices, elementwise operations such as sqrt, log, exp and pow operations as well. Tomorrow my plan is to complete the small set of autograd left -> permute matrices, layer normalization, MSE Loss and cross entropy loss and then we will create our very own neural layers!!!! Very exciting.

**Day 6- Layer Norm and permutation operation autograd**   
Added autograd for Layer normalization but there are some issues in backward propogration for this method, so I will need to look into it. Also completed the implementation of permutation of matrices autograd. Tomorrow completion with MSE Loss and cross entropy.

**Day 7 - Fixed Layer Norm**
Fixed the backpropagation of layer normalization, the issue was that the gradient of the result will of shape nxm - but the operands that achieved the result might be of shape nxm and 1xm, so the gradients for the two operands should be different, but initially we were considering the shapes of the two operands as the same which was causing repeated addition. To resolve this we created a unbroadcasting function that converts the gradient back to the operand shape (1xm) so that there is no repeated operation being performed. This took a large chunk of my time, may be later(tmrw/day after) I may complete the loss functions - I will largely focus on cross entropy as creating a language based transformer we are more focused on alphabet classification over regressive analysis. Maybe later we may implement MSE and MAD as well.

**Day I dont know**
I got fed up with the errors and gave up, now back at it, tried to fix most of it and officially I would like to announce ***I, a fucking programming noob*** have successfully implemented generic pytorch from scratch in C++, wooooooooooooooooooo.

**Day I dont know + 2**
I fixed further issues and added some mini test-cases lazy to update it, will update it later

## Neural Networks (C++)

**Day 1**
Created my first neural layers in C++ - Linear Layer, and some activation layers like Relu, GLU, ELU, Tanh, Sigmoid. This is very cool, after implementation of all the neural networks, my wish is to port it to python and develop a small application in python for using the model.

**Day 2**  
Created the Gated Residual Network and it works wahoo. I have created my first proper deep learning neural network without the help of AI, all me and it was fucking hard, I hate C++ from my very core, it is extremely hard, I don't understand vtables, referencing seg and core faults. Someone save me from this hellhole.

**Day 3**  
I forgot to write my updates yesterday, so I am writing all my updates today, I decided today to be a simple bug fixing session on my GRN network. Fixed bug issues related to vtables (Had to properly define virtual functions in my NN layers). Fixed segmentation faults in my model (I don't know what fixed it but it magically got fixed), and now I have a working GRN network to do jack shit.  Next, I will try to work in the Variable Selection Network in fucking C++. ;(

**Day 4**  
Create the Variable Selection Network, the network works fine at least mathematically for 3D tensors but for some reasons does not work for 2D tensors due to some shape handling issue. Need to look into that. Overall couple more layers left and we will port the c++ librarires to python to build a useable application for my library.

**Day 5**  
Fixed Calculation and parameter exposition in VSN and fixed static encoder. There are still issue like vanishing and exploding values that needs fixing, will see if the issues fixes itself once I implement the optimizers and if convergences will handle it or not. Otherwise, I will spend another eternity in fixing that issue.

## Backtesting Engine

Designed and implemented the foundation of a custom backtesting engine intended for seamless integration with the ARLM, enabling direct evaluation of strategy outputs generated from financial text analysis.










