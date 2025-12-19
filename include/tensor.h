#ifndef TENSOR_H
#define TENSOR_H

#include<iostream>
#include<vector>
#include<optional>
#include<algorithm>
#include<cmath>
#include<limits>
#include<numeric>
#include<numbers>
#include<omp.h>
#include<cblas.h>
#include "nditerator.h"

class Tensor {
    private:
        std::vector<double> valvec;
        std::vector<size_t> shapes;
        std::vector<size_t> strides;
        double* basePtr;
        size_t dim;

        //generalized elementwise arithematic operations
        template<typename TensorOp>
        Tensor tensorOp(const Tensor& t, TensorOp op){
            Tensor a = (t.ndim() > dim) ? singleton_rule(t) : *this;
            Tensor b = (t.ndim() < dim) ? singleton_rule(t) : t;

            std::vector<size_t> ans_shape = broadcast_shape(t);
            size_t total_size = std::accumulate(ans_shape.begin(), ans_shape.end(), size_t{1}, std::multiplies<size_t>());
            std::vector<double> new_valvec(total_size);
            std::vector<std::vector<size_t>> indices;
            for(const auto& idx: NDRange(ans_shape)){
                indices.push_back(idx);
            }

            #pragma omp parallel for if(total_size> 10000)
            for(size_t vidx=0; vidx<indices.size(); vidx++){
                const auto& idx = indices[vidx];

                std::vector<size_t> idx_a(a.ndim()), idx_b(b.ndim());

                size_t o_a = ans_shape.size() - a.ndim();
                size_t o_b = ans_shape.size() - b.ndim();

                for(size_t i=0; i<a.ndim(); i++){
                    idx_a[i] = (a.shapes[i] == 1) ? 0 : idx[i+o_a];
                }

                for(size_t i=0; i<b.ndim(); i++){
                    idx_b[i] = (b.shapes[i] == 1) ? 0 : idx[i+o_b];
                }

                double left = a.at(idx_a), right = b.at(idx_b);
                new_valvec[vidx] = op(left, right);
            }

            return Tensor(new_valvec, ans_shape);
        }

        //generalized scalar matrix operations;
        template<typename TensorOp>
        Tensor scalarOp(double val, TensorOp op){
            size_t total_size = std::accumulate(shapes.begin(), shapes.end(), size_t{1}, std::multiplies<size_t>());
            std::vector<double> new_valvec(total_size);
            std::vector<std::vector<size_t>> indices;
            for(const auto& idx: NDRange(shapes)){
                indices.push_back(idx);
            }

            #pragma omp parallel for if(total_size>10000)
            for(size_t vidx=0; vidx<total_size; vidx++){
                new_valvec[vidx] = op(at(indices[vidx]), val);
            }

            return Tensor(new_valvec, shapes); 
        }

        //reduced axis shapes and vectors for reduction operations such as sum, mean and max along axes
        std::tuple<std::vector<size_t>, std::vector<size_t>> axis_reduction(const size_t axis);

    protected:
        size_t jumpTo(std::vector<size_t> pos);
        std::vector<size_t> computeStrides(std::vector<size_t> shps);

    public:
        Tensor() = default;
        Tensor(std::vector<size_t> shape_list);
        Tensor(std::vector<double> vec, std::vector<size_t> shape_list);
        Tensor(double* ptr, std::vector<size_t> shape_list, std::vector<size_t> strides);
        Tensor(std::vector<double> vec, std::vector<size_t> shape_list, std::vector<size_t> _strides);

        //copy constructor
        Tensor(const Tensor& other);
        Tensor& operator= (const Tensor& other);

        double* data();
        const double* data() const;

        std::vector<double> as_vector();
        size_t ndim() const;
        std::vector<size_t> get_strides() const;
        std::vector<size_t> shape() const;
        size_t size() const;
        bool empty() const;

//region broadcasting rules
        bool shape_check(std::vector<size_t> t_shp);
        std::vector<size_t> broadcast_shape(const Tensor& t);
        Tensor singleton_rule(const Tensor& t);
        Tensor unsqueeze(size_t axis);
        Tensor expand(std::vector<size_t> target);
        Tensor concatenate(const Tensor& b, const size_t axis);
        Tensor mask_filled(std::vector<bool> mask, double replace);

//region access and modification
        double at(std::vector<size_t> pos);
        void put(std::vector<size_t> pos, double val);
//endregion access and modification

//region data-viewing
        // referenced slicing -> the slice is still pointing to the same location as the og tensor
        Tensor slice(std::vector<size_t> start, std::vector<size_t> shape, const std::optional<std::vector<size_t>>& _strides = std::nullopt);
        Tensor reshape(std::vector<size_t> new_shape);
        Tensor permute(const std::optional<std::vector<size_t>>& rotaxis = std::nullopt);
        Tensor transpose();
//endregion data-viewing

//region element-wise operations
        Tensor operator+ (const Tensor& t);
        Tensor operator+ (double val);
        Tensor operator- (const Tensor& t);
        Tensor operator- (double val);
        Tensor operator* (const Tensor& t) ;
        Tensor operator* (double val) ;
        Tensor operator/ (double val) ;
        Tensor operator/ (const Tensor& t);
//endregion element-wise operations

//region reductions
        Tensor sum(const size_t axis);
        Tensor mean(const size_t axis);
        Tensor maximum(const size_t axis);
//endregion reductions

        //functional operations

        // Softmax function
        Tensor softmax(const size_t axis);
        // Layer Normalization
        Tensor layer_norm(const size_t gamma, const size_t beta, const size_t axis);
        Tensor relu();
        Tensor gelu();

        //test operations
        void show();
        void prnt(std::vector<size_t> x);
        void prntd(std::vector<double> x);
};

Tensor dot(Tensor x, Tensor y, const size_t axis);

#endif