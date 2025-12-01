#ifndef NDITERATOR_H
#define NDITERATOR_H

#include<iostream>
#include<vector>

using namespace std;

class NDiterator{
    private:
        vector<size_t> shape;
        vector<size_t> current_shape;
        bool is_end;

    public:
        NDiterator(const vector<size_t>& _shape, bool end = false)
            : shape(_shape), current_shape(shape.size(), 0), is_end(end)
        {
            if(!end){
                for(size_t dim: shape){
                    if(dim == 0) {
                        is_end = true;
                        break;
                    }
                }
            }
        }

        const vector<size_t>& operator*() const { return current_shape; }
        NDiterator& operator++(){
            size_t carry = 1;
            size_t n = current_shape.size();
            for(size_t i = 0; i<current_shape.size(); i++){
                current_shape[n-i-1]++;
                if(current_shape[n-i-1]<shape[n-i-1]){
                    return *this;
                }
                current_shape[n-i-1] = 0;
            }

            is_end = true;
            return *this;
        }

        bool operator !=(const NDiterator& iter) {
            if(is_end && iter.is_end) return false;
            if(is_end != iter.is_end) return true;
            return current_shape != iter.current_shape;
        }

        static NDiterator begin(const vector<size_t>& shape){
            return NDiterator(shape, false);
        }

        static NDiterator end(const vector<size_t>& shape){
            return NDiterator(shape, true);
        }
};

class NDRange {
    private:
        vector<size_t> shape;
    public:
        NDRange(vector<size_t>& _shape) : shape(_shape) {}

        NDiterator begin() {
            return NDiterator::begin(shape);
        }

        NDiterator end() {
            return NDiterator::end(shape);
        }

        size_t size(){
            size_t sz = accumulate(shape.begin(), shape.end(), size_t{1}, multiplies<size_t>());
            return sz;
        }
};

#endif