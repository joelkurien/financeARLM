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

## Backtesting Engine

Designed and implemented the foundation of a custom backtesting engine intended for seamless integration with the ARLM, enabling direct evaluation of strategy outputs generated from financial text analysis.
