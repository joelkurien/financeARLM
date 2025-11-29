#ifndef TENSOR_H
#define TENSOR_H

#include<iostream>
#include<vector>
#include<optional>
#include<limits>
#include<numeric>

using namespace std;

class Tensor {
    private:
        vector<double> valvec;
        vector<size_t> shapes;
        vector<size_t> strides;
        double* basePtr;
        size_t dim;

    protected:
        size_t jumpTo(vector<size_t> pos){
            if(pos.size() != dim) throw "Invalid size";
            size_t val_pos = 0;
            for(size_t i=0; i<dim; i++){
                val_pos += strides[i] * pos[i];
            }
            return val_pos;
        }

        vector<size_t> computeStrides(vector<size_t> shps){
            size_t p = 1;
            vector<size_t> std(dim);
            for(int i=dim-1; i>=0; i--){
                std[i] = p;
                p*=shps[i];
            }
            return std;
        }

    public:
        Tensor() = default;

        Tensor(vector<size_t> shape_list)
        {
            shapes = shape_list;
            dim = shapes.size();
            strides = computeStrides(shapes);
            valvec.resize(size(),0);
            basePtr = valvec.data();
        }

        Tensor(double* ptr, vector<size_t> shape_list, vector<size_t> strides)
            : basePtr(ptr), strides(strides), shapes(shape_list), dim(shapes.size()) {}

        vector<size_t> shape(){ return shapes; }

        double at(vector<size_t> pos){
            size_t val_pos = jumpTo(pos);
            return basePtr[val_pos];
        }

        void put(vector<size_t> pos, double val) {
            size_t idx = jumpTo(pos);
            if(idx < size()) basePtr[idx] = val;
        }

        size_t size() const { return accumulate(shapes.begin(), shapes.end(), size_t{1}, multiplies<size_t>()); }

        bool empty(){ return (dim == 0 ? true : false); }

        // referenced slicing -> the slice is still pointing to the same location as the og tensor
        Tensor slice(vector<size_t> start, vector<size_t> shape, const optional<vector<size_t>>& _strides = nullopt){
            vector<size_t> actualStrides = _strides.value_or(strides);
            double* subBasePtr = basePtr + jumpTo(start);
            return Tensor(subBasePtr, shape, actualStrides);
        }

        Tensor reshape(vector<size_t> new_shape){
            size_t p = accumulate(new_shape.begin(), new_shape.end(), size_t{1}, multiplies<size_t>());
            if(size() != p) return Tensor();
            vector<size_t> new_strides = computeStrides(new_shape);
            return Tensor(basePtr, new_shape, new_strides);
        }

        Tensor permute(const optional<vector<size_t>>& rotaxis = nullopt) {
            vector<size_t> new_stride = strides;
            vector<size_t> new_shape = shapes;
            if(rotaxis != nullopt && rotaxis->size() == dim){
                for(size_t i=0; i<dim; i++){
                    new_stride[i] = strides[rotaxis->at(i)];
                    new_shape[i] = shapes[rotaxis->at(i)];
                }
            } else {
                reverse(new_stride.begin(), new_stride.end());
                reverse(new_shape.begin(), new_shape.end());
            }

            return Tensor(basePtr, new_shape, new_stride);  
        }

        void show(){
            for(int i=0; i<size(); i++)
                cout<<*(basePtr+i)<<" ";
        }
};

#endif