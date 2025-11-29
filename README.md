* Implementing a transformer from scratch and using it to build an ARLM for summarizing or extracting key information from financial papers and using it to train financial strategies
  **Numpy Tensor Implementations**
    - Day 1: Created the C++ Executables using CMake and installed and integrated the Eigen Library for matrix implementation and linear algebra operations.
    - Day 2: Created the Tensor library -> Focused more on 2D tensors, created shape functions, update operation, memory stride views for memory efficient matrix updation.
    - Day 3: Created the 3D tensors for batched tensor processing (heavily used in transformers) - implemented copy batch slicing, base batch operations
    - Day 4: Generalized the two tensor classes into a single tensor that accessed the tensor data using pointer + shape arithematics
          - Implemented view-based slicing, reshaping; optimized stride calculations
    - Day 5: Added Broadcasting rules such as shape checks, stride updations for unequal dimensions, unsqueeze and expand dimensions rules + Added the boilerplate for arithematic operations
* Building a backtesting engine from scratch to integrate with the ARLM 
