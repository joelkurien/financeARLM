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

## Backtesting Engine

Designed and implemented the foundation of a custom backtesting engine intended for seamless integration with the ARLM, enabling direct evaluation of strategy outputs generated from financial text analysis.



