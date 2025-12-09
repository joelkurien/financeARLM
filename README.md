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

**Day 9 - Matrix multiplication and broadcasting rules
Implemented matrix multiplication using BLAS library because it was too complex and time consuming. I had to refer multiple sources on how to handle concurrency and control GPU/CPU performance so that I dont burn out my CPU, but failed miserably so I just resorted to using BLAS because I am lazy af.

**Day 10 - Broadcasting rules**
Completed all required broadcasting rules that are necessary to build a transformer/basic feedforward network. Further implemented the Relu and Gelu functions as well. We have completed all the necessary functions required by a tensor library to build transformers and neural networks. Fucking hell!!!! I fucking hate C++ but have to do it if I want to get a good career somewhere. 

## Autograd Library (C++)

## Backtesting Engine

Designed and implemented the foundation of a custom backtesting engine intended for seamless integration with the ARLM, enabling direct evaluation of strategy outputs generated from financial text analysis.
