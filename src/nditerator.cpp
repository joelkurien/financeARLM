#include "nditerator.h"
#include<numeric>

NDiterator::NDiterator(const std::vector<size_t>& _shape, bool end)
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

const std::vector<size_t>& NDiterator::operator*() const { return current_shape; }

NDiterator& NDiterator::operator++() {
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

bool NDiterator::operator !=(const NDiterator& iter) {
    if(is_end && iter.is_end) return false;
    if(is_end != iter.is_end) return true;
    return current_shape != iter.current_shape;
}

NDiterator NDiterator::begin(const std::vector<size_t>& shape){
    return NDiterator(shape, false);
}

NDiterator NDiterator::end(const std::vector<size_t>& shape){
    return NDiterator(shape, true);
}

NDRange::NDRange(std::vector<size_t>& _shape) : shape(_shape) {}

NDiterator NDRange::begin() {
    return NDiterator::begin(shape);
}

NDiterator NDRange::end() {
    return NDiterator::end(shape);
}

size_t NDRange::size(){
    size_t sz = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
    return sz;
}
