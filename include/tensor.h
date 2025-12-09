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
        Tensor tensorOp(Tensor& t, TensorOp op){
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
                double left = a.at(indices[vidx]), right = b.at(indices[vidx]);
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
        std::tuple<std::vector<size_t>, std::vector<size_t>> axis_reduction(const size_t axis){
            if(axis >= dim) {
                    throw std::invalid_argument("Invalid axis value");
            }
            std::vector<size_t> reduced_dim;
            for(size_t i=0; i<dim; i++){
                if(i != axis){
                    reduced_dim.push_back(i);
                }
            }

            std::vector<size_t> reduced_shape;
            for(size_t i=0; i<reduced_dim.size(); i++){
                reduced_shape.push_back(shapes[reduced_dim[i]]);
            }

            auto indices = NDRange(reduced_shape);
            
            std::vector<size_t> base_idx;
            for(const auto& idx: indices){
                size_t base = 0;
                for(size_t i=0;i<reduced_dim.size(); i++){
                    base += idx[i]*strides[reduced_dim[i]];
                }
                base_idx.push_back(base);
            }
            return {base_idx, reduced_shape};
        }

    protected:
        size_t jumpTo(std::vector<size_t> pos){
            if(pos.size() != dim) throw std::invalid_argument("Invalid size");
            size_t val_pos = 0;
            for(size_t i=0; i<dim; i++){
                val_pos += strides[i] * pos[i];
            }
            return val_pos;
        }

        std::vector<size_t> computeStrides(std::vector<size_t> shps){
            size_t p = 1;
            std::vector<size_t> std(dim);
            for(int i=dim-1; i>=0; i--){
                std[i] = p;
                p*=shps[i];
            }
            return std;
        }

    public:
        Tensor() = default;

        Tensor(std::vector<size_t> shape_list)
        {
            if(shape_list.size() < 2) throw std::invalid_argument("Vector shape should have at least 2 dimensions");
            shapes = shape_list;
            dim = shapes.size();
            strides = computeStrides(shapes);
            valvec.resize(size(),0);
            basePtr = valvec.data();
        }

        Tensor(std::vector<double> vec, std::vector<size_t> shape_list)
            : shapes(shape_list), dim(shapes.size()), valvec(vec)
        {
            basePtr = valvec.data();
            strides = computeStrides(shapes);
        }

        Tensor(double* ptr, std::vector<size_t> shape_list, std::vector<size_t> strides)
            : basePtr(ptr), strides(strides), shapes(shape_list), dim(shapes.size()) {}

        double* data() { return basePtr; }
        const double* data() const { return basePtr; }

        std::vector<double> as_vector() { return valvec; }
        size_t ndim() const { return dim; }
        std::vector<size_t> get_strides() const { return strides; }    
        std::vector<size_t> shape() const { return shapes; }

        size_t size() const { return std::accumulate(shapes.begin(), shapes.end(), size_t{1}, std::multiplies<size_t>()); }

        bool empty() const { return (dim == 0 ? true : false); }

//region broadcasting rules
        bool shape_check(std::vector<size_t> t_shp){
            size_t tdim = t_shp.size();
            size_t maxl = std::max(tdim, dim);

            std::vector<size_t> smv(maxl,1);
            const auto& src = tdim >= dim ? shapes : t_shp;
            const auto& trg = tdim >= dim ? t_shp : shapes;
            size_t nd = tdim >= dim ? dim : tdim;
            for(size_t i=0; i<nd; i++){
                smv[maxl-1-i] = src[nd-1-i];
            }

            for(size_t i=0; i<maxl; i++)
                if(!(smv[maxl-i-1] == trg[maxl-i-1] || smv[maxl-i-1] == 1 || trg[maxl-i-1] == 1)) throw std::runtime_error("Shapes incompatible");
            return true;
        }

        std::vector<size_t> broadcast_shape(Tensor& t){
            if(!shape_check(t.shape())) {}
            std::vector<size_t> fnl_shape;
            size_t maxl = std::max(t.ndim(), dim);

            std::vector<size_t>smv(maxl,1);
            const auto& src = t.ndim() >= dim ? shapes : t.shape();
            const auto& trg = t.ndim() >= dim ? t.shape() : shapes;
            size_t nd = t.ndim() >= dim ? dim : t.ndim();
            for(size_t i=0; i<nd; i++){
                smv[maxl-1-i] = src[nd-1-i];
            }

            for(size_t i=0; i<maxl; i++)
                smv[maxl-i-1] = std::max(smv[maxl-i-1], trg[maxl-i-1]); 
            fnl_shape = smv;  
            return fnl_shape;
        }

        Tensor singleton_rule(Tensor& t){
            if(!shape_check(t.shape())) {}
            std::vector<size_t> fnl_shape;
            size_t maxl = std::max(t.ndim(), dim);

            std::vector<size_t>smv(maxl,1);
            const auto& src = t.ndim() >= dim ? shapes : t.shape();
            const auto& trg = t.ndim() >= dim ? t.shape() : shapes;
            double* ptr = t.ndim() >= dim ? basePtr : t.data();
            std::vector<size_t> newStrides = t.ndim() >= dim ? get_strides() : t.get_strides();
            size_t nd = t.ndim() >= dim ? dim : t.ndim();
            for(size_t i=0; i<nd; i++){
                smv[maxl-1-i] = src[nd-1-i];
            }
            newStrides.insert(newStrides.begin(), maxl-nd, 0);
            for(size_t i=0; i<maxl; i++){
                if(trg[maxl - i - 1]>smv[maxl - i - 1]){
                    smv[maxl-i-1] = trg[maxl-i-1];
                    newStrides[maxl-i-1] = 0;
                }
            }
            
            return Tensor(ptr, smv, newStrides);
        }

        Tensor unsqueeze(size_t axis){
            std::vector<size_t> new_shape = shapes;
            std::vector<size_t> new_strides = strides;
            new_shape.insert(new_shape.begin()+axis, 1);
            new_strides.insert(new_strides.begin()+axis, 0);
            return Tensor(basePtr, new_shape, new_strides);
        }

        Tensor expand(std::vector<size_t> target){
            std::vector<size_t> new_strides = strides;
            if(shape_check(target)){
                for(size_t i=0; i<target.size(); i++){
                    if(shapes[i] > target[i]) throw std::runtime_error("Target shape should be greater");
                    if(shapes[i] == 1 && target[i] == 1) continue;
                    if(shapes[i] == 1) new_strides[i] = 0;
                }
            }
            return Tensor(basePtr, target, new_strides);
        }

        Tensor concatenate(Tensor& b, const size_t axis){
            if(dim != b.ndim()) throw std::invalid_argument("Tensor dimension mismatch");
            if(axis >= dim) throw std::invalid_argument("Axis is invalid");
            std::vector<size_t> new_shape(dim,0);
            for(int i=0; i<dim; i++){
                if(i != axis && shapes[i] != b.shapes[i]){
                    throw std::invalid_argument("Tensor non-target axis value mismatch");
                } 
                new_shape[i] = (i == axis) ? shapes[axis]+b.shapes[axis] : shapes[i];
            }

            Tensor conc(new_shape);

            size_t left_size = 1;
            for(size_t i=0; i<axis; i++)
                left_size *= shapes[i];
            
            size_t a_size = 1;
            for(size_t i = axis; i<dim; i++){
                a_size *= shapes[i];
            }

            size_t b_size = 1;
            for(size_t i = axis; i<dim; i++){
                b_size *= b.shapes[i];
            }

            size_t conc_idx = 0;
            for(size_t c_idx = 0; c_idx < left_size; c_idx++){
                size_t a_lft = c_idx*a_size;
                size_t b_lft = c_idx*b_size;

                std::copy(valvec.begin()+a_lft, valvec.begin() + a_lft + a_size, conc.valvec.begin()+conc_idx);
                conc_idx += a_size;

                std::copy(valvec.begin()+b_lft, valvec.begin()+b_lft+b_size, conc.valvec.begin()+conc_idx);
                conc_idx += b_size;
            }
            // vector<size_t> left (shapes.begin(), shapes.begin()+axis);

            // vector<size_t> r1 (shapes.begin()+axis, shapes.end());
            // vector<size_t> r2 (b.shapes.begin()+axis, b.shapes.end());
            // auto l_indices = NDRange(left);
            // auto a_indices = NDRange(r1);
            // auto b_indices = NDRange(r2);

            // Tensor conc (new_shape);
            // size_t index = 0;
            // for(const vector<size_t>& l_idx: l_indices){
            //     vector<size_t> pos (dim, 0);

            //     for(size_t i=0; i<l_idx.size(); i++){
            //         pos[i] = l_idx[i];
            //     }
            //     for(const vector<size_t>& idx: a_indices){
            //         for(size_t x=0; x<idx.size(); x++){
            //             pos[axis+x] = idx[x];
            //         }
            //         conc.valvec[index++] = at(pos);
            //     }
            //     for(const vector<size_t>& idx: b_indices){
            //         for(size_t x=0; x<idx.size(); x++){
            //             pos[axis+x] = idx[x];
            //         }
            //         conc.valvec[index++] = b.at(pos);
            //     }
            // }
            return conc;
        }

        Tensor mask_filled(std::vector<bool> mask, double replace){
            if(mask.size() != size()) throw std::invalid_argument("Mask/Matrix size mismatch");
            std::vector<double> n_vec;
            n_vec.reserve(size());
            for(int i=0; i<size(); i++){
                n_vec.push_back(mask[i] ? replace : valvec[i]);
            }
            return Tensor(n_vec, shapes);
        }

//region access and modification
        double at(std::vector<size_t> pos){
            size_t val_pos = jumpTo(pos);
            return basePtr[val_pos];
        }

        void put(std::vector<size_t> pos, double val) {
            size_t idx = jumpTo(pos);
            if(idx < size()) basePtr[idx] = val;
        }
//endregion access and modification

//region data-viewing
        // referenced slicing -> the slice is still pointing to the same location as the og tensor
        Tensor slice(std::vector<size_t> start, std::vector<size_t> shape, const std::optional<std::vector<size_t>>& _strides = std::nullopt){
            std::vector<size_t> actualStrides = _strides.value_or(strides);
            double* subBasePtr = basePtr + jumpTo(start);
            return Tensor(subBasePtr, shape, actualStrides);
        }

        Tensor reshape(std::vector<size_t> new_shape){
            size_t p = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>());
            if(size() != p) return Tensor();
            std::vector<size_t> new_strides = computeStrides(new_shape);
            return Tensor(basePtr, new_shape, new_strides);
        }

        Tensor permute(const std::optional<std::vector<size_t>>& rotaxis = std::nullopt) {
            std::vector<size_t> new_stride = strides;
            std::vector<size_t> new_shape = shapes;
            if(rotaxis != std::nullopt && rotaxis->size() == dim){
                for(size_t i=0; i<dim; i++){
                    new_stride[i] = strides[rotaxis->at(i)];
                    new_shape[i] = shapes[rotaxis->at(i)];
                }
            } else {
                std::reverse(new_stride.begin(), new_stride.end());
                std::reverse(new_shape.begin(), new_shape.end());
            }

            return Tensor(basePtr, new_shape, new_stride);  
        }
//endregion data-viewing

//region element-wise operations
        Tensor operator+ (Tensor& t) {
            return tensorOp(t, [](double a, double b){ return a+b; });
        }
        
        Tensor operator+ (double val){
            return scalarOp(val, [](double a, double b){ return a+b; }); 
        }

        Tensor operator- (Tensor& t){
            return tensorOp(t, [](double a, double b){ return a-b; });
        }

        Tensor operator- (double val) {
            return scalarOp(val, [](double a, double b){ return a-b; });
        }

        Tensor operator* (Tensor& t) {
            return tensorOp(t, [](double a, double b){ return a*b; });
        }

        Tensor operator* (double val) {
             return scalarOp(val, [](double a, double b){ return a*b; });
        }

        Tensor operator/ (double val) {
            return scalarOp(val, [](double a, double b){ return a/b; });
        }

        Tensor operator/ (Tensor& t){
            return tensorOp(t, [](double a, double b){ return a/b; });
        }
//endregion element-wise operations

//region reductions
        Tensor sum(const size_t axis){
            auto [base_idx, reduced_shape] = axis_reduction(axis);
            std::vector<double> result(base_idx.size());

            #pragma omp parallel for
            for(size_t i=0; i<base_idx.size(); i++){
                double sum = 0;
                for(size_t j=0; j<shapes[axis]; j++){
                    sum += basePtr[base_idx[i] + strides[axis]*j];
                }
                result[i] = sum;
            }
            return Tensor(result, reduced_shape);
        }

        Tensor mean(const size_t axis){
            auto [base_idx, reduced_shape] = axis_reduction(axis);
            std::vector<double> result(base_idx.size());

            #pragma omp parallel for
            for(size_t i=0; i<base_idx.size(); i++){
                double sum = 0;
                for(size_t j=0; j<shapes[axis]; j++){
                    sum += basePtr[base_idx[i] + strides[axis]*j];
                }
                result[i] = sum/shapes[axis];
            }
            return Tensor(result, reduced_shape);
        }

        Tensor maximum(const size_t axis){
            auto [base_idx, reduced_shape] = axis_reduction(axis);
            std::vector<double> result(base_idx.size());

            #pragma omp parallel for
            for(size_t i=0; i<base_idx.size(); i++){
                double mx = -std::numeric_limits<double>::infinity();
                for(size_t j=0; j<shapes[axis]; j++){
                    double val = basePtr[base_idx[i] + strides[axis]*j];
                    mx = mx < val ? val : mx;
                }
                result[i] = mx;
            }
            return Tensor(result, reduced_shape);
        }
//endregion reductions

        //functional operations

        // Softmax function
        Tensor softmax(const size_t axis){
            auto [base_idx, reduced_shape] = axis_reduction(axis);
            std::vector<double> result(size());

            const size_t sax = shapes[axis];
            const size_t stride = strides[axis];

            #pragma omp parallel for
            for(size_t i=0; i<base_idx.size(); i++){
                double mx = -std::numeric_limits<double>::infinity();
                for(size_t j=0; j<sax; j++){
                    double val = basePtr[base_idx[i] + stride*j];
                    mx = mx < val ? val : mx;
                }
                double denom = 0;
                for(size_t j=0; j<sax; j++){
                    denom += exp(basePtr[base_idx[i] + stride*j] - mx);
                }
                for(size_t j=0; j<sax; j++){
                    size_t idx = base_idx[i] + stride*j;
                    result[idx] = exp(basePtr[idx] - mx) / denom;
                }
            }
            return Tensor(result, shapes);
        }

        // Layer Normalization
        Tensor layer_norm(const size_t gamma, const size_t beta, const size_t axis){
            auto [base_idx, reduced_shape] = axis_reduction(axis);
            std::vector<double> result(size());

            const size_t sax = shapes[axis];
            const size_t stride = strides[axis];

            #pragma omp parallel for
            for(size_t i=0; i<base_idx.size(); i++){
                double mu = 0, var = 0;
                for(size_t j=0; j<sax; j++){
                    mu += basePtr[base_idx[i] + stride*j];
                }
                mu /= sax;
                for(size_t j=0; j<sax; j++){
                    var += pow((basePtr[base_idx[i] + stride*j]-mu),2);
                }
                var /= sax;
                double e = 1e-12;
                double inv_std = 1.0 / sqrt(var + e);
                for(size_t j=0; j<sax; j++){
                    size_t idx = base_idx[i] + stride*j;
                    result[idx] = gamma * ((basePtr[idx] - mu) * inv_std ) + beta;
                }
            }
            return Tensor(result, shapes);
        }

        Tensor relu(){
            std::vector<double> res;
            res.reserve(size());

            for(int i=0; i<size(); i++){
                res.push_back(std::max(valvec[i], 0.0));
            }

            return Tensor(res, shapes);
        }

        Tensor gelu(){
            std::vector<double> res;
            res.reserve(size());

            const double constant = sqrt(2.0 / std::numbers::pi);
            for(int i=0; i<size(); i++){
                double x = valvec[i];
                double cube = x*x*x;
                double inner = constant*(x+0.044715*cube);
                double val = 0.5 * x * (1.0 + tanh(inner));
                res.push_back(val);
            }

            return Tensor(res, shapes);
        }

        //test operations
        void show(){
            for(int i=0; i<size(); i++)
                std::cout<<*(basePtr+i)<<" ";
        }

        void prnt(std::vector<size_t> x){
            for(auto e: x){
                std::cout<<e<<" ";
            }
            std::cout<<std::endl;
        }

        void prntd(std::vector<double> x){
            for(auto e: x){
                std::cout<<e<<" ";
            }
            std::cout<<std::endl;
        }
};

#endif