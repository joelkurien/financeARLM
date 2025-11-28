#ifndef TENSOR_H
#define TENSOR_H

#include<iostream>
#include<vector>
#include<optional>
#include<limits>

using namespace std;

class Tensor {
    private:
        vector<double> valvec;
        vector<size_t> shapes;
        vector<size_t> strides;
        double* basePtr;
        size_t dim;
        bool ownMem;

    protected:
        size_t jumpTo(vector<size_t> pos){
            if(pos.size() != dim) throw "Invalid size";
            size_t val_pos = 0, i = 0;
            for(int i=0; i<dim; i++){
                val_pos += strides[i] * pos[i];
            }
            return val_pos;
        }

        vector<size_t> computeStrides(vector<size_t> shps){
            reverse(shps.begin(), shps.end());
            size_t p = 1;
            vector<size_t> std;
            for(int i=0; i<dim; i++){
                std.push_back(p);
                p*=shps[i];
            }
            reverse(std.begin(), std.end());
            return std;
        }

    public:
        Tensor() = default;

        Tensor(initializer_list<size_t> shape_list) : ownMem(true)
        {
            size_t p = 1;
            shapes = shape_list;
            dim = shapes.size();
            strides = computeStrides(shapes);
            valvec={-1};
            basePtr = valvec.data();
        }

        Tensor(double* ptr, initializer_list<size_t> shape_list, vector<size_t> strides)
            :Tensor(shape_list)
        {
            basePtr = ptr;
            this->strides = strides;
            ownMem = false;
            dim = shapes.size();
        }

        void show(){
            if(ownMem){
                basePtr = valvec.data();
                cout<<(basePtr+3*1+1)<<endl;
            }
            else cout<<basePtr<<endl;
        }

        vector<size_t> shape(){ return shapes; }

        double at(vector<size_t> pos){
            size_t val_pos = jumpTo(pos);

            if(ownMem) basePtr = valvec.data();
            double val = *(basePtr + val_pos+1);
            return val;
        }

        void put(double val) { 
            valvec.push_back(val); 
        }

        void put(vector<size_t> pos, double val) {
            size_t val_pos = jumpTo(pos);
            if(ownMem) basePtr = valvec.data();
            *(basePtr + val_pos+1) = val;
        }

        size_t size(){
            size_t p = 1;
            for(auto sh: shapes){
                p *= sh;
            }   
            return p;
        }

        bool empty(){ return (dim == 0 ? true : false); }

        // referenced slicing -> the slice is still pointing to the same location as the og tensor
        Tensor slice(vector<size_t> start, initializer_list<size_t> shape, optional<vector<size_t>> _strides = nullopt){
            vector<size_t> actualStrides = _strides.value_or(strides);
            double* subBasePtr = valvec.data() + jumpTo(start);
            Tensor slc(subBasePtr, shape, actualStrides);
            return slc;
        }

        Tensor reshape(vector<size_t> new_shape){
            size_t p = 1;
            for(auto e: new_shape) p *= e;
            Tensor slc();
            if(size() == p ){
                double* subBasePtr = valvec.data();
                Tensor slc(subBasePtr, new_shape, computeStrides(new_shape));
            }
            return slc;
        }

        Tensor transpose() {}
};

#endif