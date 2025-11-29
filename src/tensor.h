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

        double* data() { return basePtr; }
        size_t ndim() const { return dim; }
        vector<size_t> get_strides() const { return strides; }    
        vector<size_t> shape() const { return shapes; }

        size_t size() const { return accumulate(shapes.begin(), shapes.end(), size_t{1}, multiplies<size_t>()); }

        bool empty() const { return (dim == 0 ? true : false); }

        //broadcasting rules
        bool shape_check(vector<size_t> t_shp){
            size_t tdim = t_shp.size();
            size_t maxl = max(tdim, dim);

            vector<size_t> smv(maxl,1);
            const auto& src = tdim >= dim ? shapes : t_shp;
            const auto& trg = tdim >= dim ? t_shp : shapes;
            size_t nd = tdim >= dim ? dim : tdim;
            for(size_t i=0; i<nd; i++){
                smv[maxl-1-i] = src[nd-1-i];
            }

            cout<<"YOU"<<endl;
            for(size_t i=0; i<maxl; i++)
                if(!(smv[maxl-i-1] == trg[maxl-i-1] || smv[maxl-i-1] == 1 || trg[maxl-i-1] == 1)) throw "Shapes incompatible";
            return true;
        }

        vector<size_t> broadcast_shape(Tensor& t){
            if(!shape_check(t.shape())) {}
            vector<size_t> fnl_shape;
            size_t maxl = max(t.ndim(), dim);

            vector<size_t>smv(maxl,1);
            const auto& src = t.ndim() >= dim ? shapes : t.shape();
            const auto& trg = t.ndim() >= dim ? t.shape() : shapes;
            size_t nd = t.ndim() >= dim ? dim : t.ndim();
            for(size_t i=0; i<nd; i++){
                smv[maxl-1-i] = src[nd-1-i];
            }

            for(size_t i=0; i<maxl; i++)
                smv[maxl-i-1] = max(smv[maxl-i-1], trg[maxl-i-1]); 
            fnl_shape = smv;  
            return fnl_shape;
        }

        Tensor singleton_rule(Tensor& t){
            if(!shape_check(t.shape())) {}
            vector<size_t> fnl_shape;
            size_t maxl = max(t.ndim(), dim);

            vector<size_t>smv(maxl,1);
            const auto& src = t.ndim() >= dim ? shapes : t.shape();
            const auto& trg = t.ndim() >= dim ? t.shape() : shapes;
            double* ptr = t.ndim() >= dim ? basePtr : t.data();
            vector<size_t> newStrides = t.ndim() >= dim ? get_strides() : t.get_strides();
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
            vector<size_t> new_shape = shapes;
            vector<size_t> new_strides = strides;
            new_shape.insert(new_shape.begin()+axis, 1);
            new_strides.insert(new_strides.begin()+axis, 0);
            return Tensor(basePtr, new_shape, new_strides);
        }

        Tensor expand(vector<size_t> target){
            vector<size_t> new_strides = strides;
            if(shape_check(target)){
                for(size_t i=0; i<target.size(); i++){
                    if(shapes[i] > target[i]) throw "Target shape should be greater";
                    if(shapes[i] == 1 && target[i] == 1) continue;
                    if(shapes[i] == 1) new_strides[i] = 0;
                }
            }
            return Tensor(basePtr, target, new_strides);
        }


//         3. Concatenate - Join tensors along an axis
// This combines multiple tensors along an existing dimension.
// Example: In multi-head attention, each head produces output [batch, seq_len, head_dim]. You have 8 heads, so you concatenate them along the last dimension to get [batch, seq_len, 8*head_dim] which equals [batch, seq_len, d_model]. This is also critical for KV caching during generation where you concatenate new keys/values with previously cached ones along the sequence dimension.

// 4. Masked Fill - Replace values where condition is true
// This fills positions in a tensor with a specified value wherever a mask is true (or non-zero).
// Example: You have attention scores [batch, heads, seq_q, seq_k] and a mask of the same shape. Wherever the mask is 0 (masked position), you fill the attention score with -infinity. This ensures those positions get zero attention weight after softmax, effectively blocking the model from attending to future tokens or padding tokens.       
        //access and modification
        double at(vector<size_t> pos){
            size_t val_pos = jumpTo(pos);
            return basePtr[val_pos];
        }

        void put(vector<size_t> pos, double val) {
            size_t idx = jumpTo(pos);
            if(idx < size()) basePtr[idx] = val;
        }

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

        //arithematic operations
        Tensor operator+ (Tensor& t) {
            Tensor max_tensor = t.ndim() >= dim ? t : *this;
            Tensor s = singleton_rule(t);
            vector<size_t> s_strides = s.get_strides();
            vector<size_t> ans_shape = broadcast_shape(t);
            
        }
        
        // Tensor operator- (Tensor& t) {}

        // Tensor operator* (Tensor& t) {}

        // Tensor operator* (double val) {}

        // Tensor operator/ (Tensor& t) {} 

        // Tensor operator/ (Tensor& t) {}

        //functional operations

        //test operations
        void show(){
            for(int i=0; i<size(); i++)
                cout<<*(basePtr+i)<<" ";
        }

        void prnt(vector<size_t> x){
            for(auto e: x){
                cout<<e<<" ";
            }
            cout<<endl;
        }
};

#endif